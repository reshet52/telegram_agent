"""Complete, source-audited reply data. No sampling cap or runtime-data migration."""

import math
import random
from collections import Counter
from uuid import uuid4

from mybot.ai.local_format import FORMAT_VERSION, training_tokens
from mybot.services.conversation_boundaries import gap_seconds, sessions
from mybot.services.history_training import tokenizer_hash
from mybot.services.learning_dataset import encoded, example, sha
from mybot.services.reviewed_dataset import reviewed_examples
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import digest, now
from mybot.storage.style_lesson import StyleLesson


def seal(data):
    return sha(encoded([{k: v for k, v in data.items() if k != 'hash'}]))


def fit(context, response, tokenizer, max_length):
    """Trim only older context; retain the complete last incoming turn and target."""
    context = list(context)
    last_incoming = len(context) - 1
    while last_incoming > 0 and context[last_incoming - 1]['sender'] != 'Я':
        last_incoming -= 1
    while context:
        row = example(context, response)
        try:
            tokens, _ = training_tokens(tokenizer, row)
        except ValueError:
            return None
        # MLX shifts labels by one: reserve one token beyond the full sequence.
        if len(tokens) < max_length:
            return context, row
        if last_incoming <= 0:
            return None
        context.pop(0)
        last_incoming -= 1
    return None


def previous_training_ids(lesson):
    used = set()
    for path in (lesson.root / 'mlx_runs').glob('*/run.json'):
        run = read_json(path, {})
        if run.get('completed_steps', 0) <= 0:
            continue
        if (run.get('account_id'), run.get('dialog_id')) != (
                lesson.workspace.account_id, lesson.workspace.dialog_id):
            raise ValueError('Чужой запуск в каталоге диалога.')
        manifest = read_json(lesson.root / 'reviewed' / (run.get('dataset_id', '') + '.json'))
        if not manifest:
            raise ValueError('Нет источников предыдущего обучения: чистоту проверки подтвердить нельзя.')
        used.update(i for item in manifest['examples']['train'] for i in item['ids'])
    return used


def prepare_full(workspace, tokenizer, model_path, *, max_length=2048):
    if max_length not in (512, 1024, 2048):
        raise ValueError('Недопустимая длина полного набора.')
    lesson = StyleLesson(workspace)
    messages = lesson.store.history()
    if len(messages) < 40:
        raise ValueError('Недостаточно истории для полной подготовки.')
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    previous = previous_training_ids(lesson)
    # Preserve every old test source, whether or not it is currently active.
    reserved = {i for path in lesson.root.glob('holdout*.json')
                for key in read_json(path, {}).get('cases', [])
                for case in [lesson.store.get(key)]
                for i in case['context_ids'] + case['expected_ids']}
    from mybot.services.reply_evaluation import reserved as evaluation_reserved
    reserved.update(evaluation_reserved(workspace))
    cutoff = messages[int(len(messages) * .95)]['message_id']
    ledger = {m['message_id']: {'message_id': m['message_id'], 'uses': [],
              'reason': 'no_complete_reply_context'} for m in messages}
    splits = {'train': [], 'valid': []}
    for session in sessions(messages, gap_seconds(workspace)):
        # Unsupported media/deletions are barriers, never silently omitted turns.
        blocks, block = [], []
        for m in session:
            reason = ('deleted' if m['message_id'] in deleted else
                      'media_without_verified_text' if m['type'] != 'text' else
                      'empty_text' if not isinstance(m['text'], str) or not m['text'].strip() else None)
            if reason:
                if block:
                    blocks.append(block)
                block = []
                ledger[m['message_id']]['reason'] = reason
            else:
                block.append(m)
        if block:
            blocks.append(block)
        for block in blocks:
            groups = []
            for m in block:
                if not groups or (groups[-1][0]['sender'] == 'Я') != (m['sender'] == 'Я'):
                    groups.append([])
                groups[-1].append(m)
            preceding = []
            for group_index, group in enumerate(groups):
                if group[0]['sender'] != 'Я' or not preceding:
                    preceding.extend(group)
                    continue
                last_incoming = groups[group_index - 1]
                context = preceding[-max(24, len(last_incoming)):]
                result = fit(context, group, tokenizer, max_length)
                original_ids = [m['message_id'] for m in context + group]
                reason = None
                if not result:
                    reason = 'target_or_last_incoming_too_long_or_invalid'
                else:
                    context, row = result
                    ids = [m['message_id'] for m in context + group]
                    split = 'valid' if min(ids) >= cutoff else 'train'
                    if min(ids) < cutoff <= max(ids):
                        reason = 'crosses_validation_boundary'
                    elif reserved.intersection(ids):
                        reason = 'reserved_old_holdout'
                    elif split == 'valid' and previous.intersection(ids):
                        reason = 'previously_trained_validation_source'
                    else:
                        item = {'case_id': 'full:' + str(group[0]['message_id']),
                                'source_kind': 'historical_sent_reply',
                                'context_ids': [m['message_id'] for m in context],
                                'response_ids': [m['message_id'] for m in group],
                                'ids': ids, 'source_digest': digest(context + group),
                                'date': group[0]['date'], 'row': row}
                        splits[split].append(item)
                        for i in item['context_ids']:
                            ledger[i]['uses'].append(split + ':context')
                        for i in item['response_ids']:
                            ledger[i]['uses'].append(split + ':target')
                        for i in set(original_ids) - set(ids):
                            ledger[i]['reason'] = 'older_context_trimmed'
                if reason:
                    for m in group:
                        ledger[m['message_id']]['reason'] = reason
                preceding.extend(group)

    reviews = reviewed_examples(workspace)
    accepted_reviews = []
    review_audit = []
    for item in reviews:
        reason = None
        if max(item['ids']) >= cutoff or reserved.intersection(item['ids']):
            reason = 'reserved_validation_source'
        else:
            case = lesson.store.get(item['case_id'])
            context, _ = lesson.store.source(case)
            result = fit(context, [{'text': item['row']['completion'][0]['content']}], tokenizer, max_length)
            if not result:
                reason = 'target_or_last_incoming_too_long_or_invalid'
            else:
                _, row = result
                accepted_reviews.append(dict(item, row=row, original_row_hash=sha(encoded([item['row']]))))
                for i in item['ids']:
                    ledger[i]['uses'].append('train:reviewed_source')
        review_audit.append({'case_id': item['case_id'], 'reason': reason or 'included'})
    # A confirmed replacement supersedes the old observed target for that turn.
    replaced = {i for item in accepted_reviews
                for i in lesson.store.get(item['case_id'])['expected_ids']}
    removed = [item for item in splits['train'] if replaced.intersection(item['response_ids'])]
    for item in removed:
        for i in item['ids']:
            role = 'train:target' if i in item['response_ids'] else 'train:context'
            ledger[i]['uses'].remove(role)
            ledger[i]['reason'] = 'superseded_by_confirmed_reply'
    splits['train'] = [item for item in splits['train'] if item not in removed] + accepted_reviews
    splits['train'].sort(key=lambda item: max(item['ids']))
    if len(splits['train']) < 8 or len(splits['valid']) < 2:
        raise ValueError('Недостаточно независимых учебных/контрольных примеров.')

    # Twenty disjoint situations, spread throughout the uncontaminated recent tail.
    independent, used = [], set()
    for item in splits['valid']:
        if not used.intersection(item['ids']):
            independent.append(item)
            used.update(item['ids'])
    if len(independent) < 20:
        raise ValueError(f'Найдено только {len(independent)} независимых новых ситуаций вместо 20.')
    chosen = [independent[round(i * (len(independent) - 1) / 19)] for i in range(20)]
    by_id = {m['message_id']: m for m in messages}
    cases = [lesson.store.create_from([by_id[i] for i in item['context_ids']],
                                     [by_id[i] for i in item['response_ids']], 'holdout')['id']
             for item in chosen]
    holdout_ref = 'holdout_' + uuid4().hex + '.json'
    lesson.write(holdout_ref, {'version': 3, 'cases': cases, 'created_at': now(),
                             'cutoff': cutoff, 'boundary_gap_seconds': gap_seconds(workspace)})
    rng = random.Random(42)
    base = list(range(len(splits['train'])))
    rng.shuffle(base)
    recent = list(range(len(splits['train']) * 2 // 3, len(splits['train'])))
    rng.shuffle(recent)
    for item in ledger.values():
        item['uses'] = sorted(set(item['uses']))
        if item['uses']:
            item['reason'] = None
    data = {'version': 3, 'id': uuid4().hex, 'mode': 'full', 'purpose': 'history',
            'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
            'created_at': now(), 'parent_adapter': None, 'format_version': FORMAT_VERSION,
            'tokenizer_hash': tokenizer_hash(model_path), 'max_length': max_length,
            'learning_rate': 2e-5, 'gap_seconds': gap_seconds(workspace),
            'holdout_ref': holdout_ref, 'cutoff': cutoff, 'fixed_holdout': True,
            'source_ids': list(by_id), 'source_digest': digest(messages),
            'examples': splits, 'counts': {k: len(v) for k, v in splits.items()},
            'order': base + recent, 'base_steps': len(base), 'reviewed_count': len(accepted_reviews),
            'review_audit': review_audit, 'coverage': list(ledger.values()),
            'coverage_counts': dict(Counter(x['reason'] or 'included' for x in ledger.values())),
            'period_counts': [sum(j // max(1, math.ceil(len(splits['train']) / 3)) == i
                                  for j in range(len(splits['train']))) for i in range(3)],
            'dates': [messages[0]['date'], messages[-1]['date']],
            'estimated_minutes': [len(base + recent) * 2 // 60, len(base + recent) * 8 // 60]}
    data['hash'] = seal(data)
    write_json(lesson.root / 'reviewed' / (data['id'] + '.json'), data)
    return data


def load_full(workspace, data):
    if data.get('hash') != seal(data) or (data.get('account_id'), data.get('dialog_id')) != (
            workspace.account_id, workspace.dialog_id):
        raise ValueError('Полный набор изменён или принадлежит другому диалогу.')
    lesson = StyleLesson(workspace)
    history = {m['message_id']: m for m in lesson.store.history()}
    if any(i not in history for i in data['source_ids']) or digest(
            [history[i] for i in data['source_ids']]) != data['source_digest']:
        raise ValueError('Исходная история изменилась; продолжение запрещено.')
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    reviews = {item['case_id']: item for item in reviewed_examples(workspace)}
    ids = {}
    for split, items in data['examples'].items():
        if len(items) != data['counts'][split]:
            raise ValueError('Неполный набор.')
        ids[split] = set()
        for item in items:
            ids[split].update(item['ids'])
            if deleted.intersection(item['ids']):
                raise ValueError('Источник обучения удалён.')
            if item['case_id'].startswith('full:'):
                context = [history[i] for i in item['context_ids']]
                response = [history[i] for i in item['response_ids']]
                good = (all(m['sender'] == 'Я' for m in response)
                        and digest(context + response) == item['source_digest']
                        and example(context, response) == item['row'])
            else:
                if item.get('review_id'):
                    evidence = lesson.read('confirmed/' + item['review_id'] + '.json')
                    case = lesson.store.get(item['case_id'])
                    context, expected = lesson.store.source(case)
                    original = {'row': example(context, [{'text': evidence['text']}]),
                                'source_digest': case['source_digest']} if evidence else None
                else:
                    original = reviews.get(item['case_id'])
                good = (original and sha(encoded([original['row']])) == item['original_row_hash']
                        and original['source_digest'] == item['source_digest'])
            if not good:
                raise ValueError('История или подтверждённые исправления изменились.')
    if ids['train'] & ids['valid'] or max(ids['train']) >= min(ids['valid']):
        raise ValueError('Обучение пересекается с проверкой.')
    from mybot.services.reply_evaluation import reserved as evaluation_reserved
    if ids['train'] & evaluation_reserved(workspace):
        raise ValueError('Учебные источники пересекаются с проверкой готовых ответов.')
    return data, {split: [item['row'] for item in items] for split, items in data['examples'].items()}

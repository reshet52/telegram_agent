"""Versioned bootstrap/replay data from real outgoing messages, not suggestions."""

import hashlib
from pathlib import Path
from uuid import uuid4

from mybot.services.history_sampling import boundaries, period, balanced
from mybot.services.learning_dataset import example, encoded, sha
from mybot.services.reviewed_dataset import holdout_examples, training_examples, action_examples
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import candidates, digest, now
from mybot.storage.style_lesson import StyleLesson
from mybot.ai.local_format import FORMAT_VERSION, training_tokens
from mybot.services.conversation_boundaries import gap_seconds, crossing_reason
from mybot.services.decision_task import training_row


def tokenizer_hash(model):
    result = hashlib.sha256()
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        path = Path(model) / name
        result.update(name.encode())
        if path.exists():
            result.update(path.read_bytes())
    return result.hexdigest()


def prepare(workspace, tokenizer, model_path, *, purpose='history', limit=512, max_length=1024):
    if purpose not in {'history', 'refine'} or not 8 <= limit <= 512:
        raise ValueError('Некорректный режим или размер набора.')
    lesson = StyleLesson(workspace)
    lesson.holdout(create=True)
    if any(item['reason'] for item in lesson.holdout_audit()):
        raise ValueError('Проверочная выборка содержит пары через паузу. Откройте предпросмотр исправленной проверки.')
    cutoff = min(lesson.reserved_ids())
    history_messages = lesson.store.history()
    options = [(c, e) for c, e in candidates(history_messages,
               load_deleted_message_ids(workspace.deleted_message_ids), set(),
               threshold=gap_seconds(workspace))
               if max(m['message_id'] for m in c + e) < cutoff]
    cuts = boundaries([e[0]['message_id'] for c, e in options])

    def fits(row):
        try:
            tokens, _ = training_tokens(tokenizer, row)
        except ValueError:
            return False
        return len(tokens) <= max_length

    historical = []
    for context, expected in options:
        row = example(context, expected)
        if not fits(row):
            continue
        historical.append({'case_id': 'history:' + str(expected[0]['message_id']),
            'source_kind': 'historical_sent_reply', 'source_digest': digest(context + expected),
            'context_ids': [m['message_id'] for m in context],
            'response_ids': [m['message_id'] for m in expected],
            'ids': [m['message_id'] for m in context + expected], 'row': row,
            'period': period(expected[0]['message_id'], cuts), 'date': expected[0].get('date')})
    parent = None
    reviews = []
    if purpose == 'refine':
        from mybot.services.style_versions import adapter, practice_state
        state = practice_state(workspace)
        parent = state.get('bootstrap') if state else None
        if not parent:
            raise ValueError('Сначала выполните первоначальное обучение по истории.')
        parent_path = adapter(workspace, parent)
        # Only replay examples actually used in the bootstrap, not newly unseen history.
        parent_run = read_json(parent_path / 'run.json')
        original = read_json(lesson.root / 'reviewed' / f"{parent_run['dataset_id']}.json")
        if original.get('holdout_ref', 'holdout.json') != lesson.holdout_ref():
            raise ValueError('Для исправленной проверки сначала обучите новую исходную версию на истории.')
        original_ids = {item['case_id'] for item in original['examples']['train']}
        reviewed = training_examples(workspace) + action_examples(workspace)
        reviewed = [item for item in reviewed if max(item['ids']) < cutoff]
        if not reviewed:
            raise ValueError('Нет подтверждённых исправлений или выбранных ответов для доработки.')
        # Omit the old target wherever the owner selected a replacement.
        replaced = {i for item in reviewed
                    if (not item['case_id'].startswith('action:')
                        or item['row']['completion'][0]['content'] == 'WAIT')
                    for i in lesson.store.get(item['case_id'].removeprefix('action:'))['expected_ids']}
        historical_actions = [dict(item, case_id='history_action:' + item['case_id'].split(':', 1)[1],
                                   source_kind='historical_observed_reply_action',
                                   row=training_row([m for m in history_messages
                                       if m['message_id'] in item['context_ids']], 'reply'))
                              for item in historical if 'history_action:' + item['case_id'].split(':', 1)[1] in original_ids]
        historical = [item for item in historical if item['case_id'] in original_ids
                      and not replaced.intersection(item['response_ids'])]
        historical += [item for item in historical_actions
                       if not replaced.intersection(item['response_ids'])]
        reviews = [item for item in reviewed if fits(item['row'])]
        if not reviews:
            raise ValueError('Подтверждённые исправления не прошли проверку длины.')
        if len(reviews) > min(256, limit):
            raise ValueError('Выбрано больше 256 исправлений: этот пробный режим требует расширения лимита, данные не отброшены.')
    reply_count = limit - len(reviews)
    # Teach the initial adapter the action task without treating silence as WAIT.
    action_count = min(reply_count // 2, max(4, reply_count // 5)) if purpose == 'history' else 0
    sampled_history = balanced(historical, reply_count - action_count)
    action_rows = []
    if purpose == 'history':
        for item in balanced(historical, action_count, seed=43):
            action_rows.append(dict(item, case_id='history_action:' + item['case_id'].split(':', 1)[1],
                                    source_kind='historical_observed_reply_action',
                                    row=training_row([m for m in history_messages
                                        if m['message_id'] in item['context_ids']], 'reply')))
    sampled = sampled_history + action_rows + reviews
    valid = [item for item in holdout_examples(workspace) if fits(item['row'])]
    if len(sampled) < 8 or len(valid) < 2:
        raise ValueError(f'После проверки длины: обучение {len(sampled)}, проверка {len(valid)}. Нужно минимум 8 и 2.')
    splits = {'train': sampled, 'valid': valid}
    dates = sorted(str(item['date']) for item in sampled if item.get('date'))
    key = uuid4().hex
    data = {'version': 2, 'id': key, 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
            'purpose': purpose, 'parent_adapter': parent, 'created_at': now(), 'fixed_holdout': True,
            'holdout_ref': lesson.holdout_ref(), 'gap_seconds': gap_seconds(workspace),
            'format_version': FORMAT_VERSION, 'learning_rate': 2e-5,
            'max_length': max_length, 'tokenizer_hash': tokenizer_hash(model_path),
            'examples': splits, 'hash': sha(encoded([splits])),
            'counts': {k: len(v) for k, v in splits.items()}, 'period_cuts': cuts,
            'source_counts': {kind: sum(item['source_kind'] == kind for item in sampled)
                              for kind in sorted({item['source_kind'] for item in sampled})},
            'period_counts': [sum(item.get('period') == i for item in sampled) for i in range(3)],
            'reviewed_count': len(reviews), 'eligible_history_count': len(historical),
            'dates': [dates[0], dates[-1]] if dates else [],
            'estimated_minutes': [max(1, len(sampled) // 60), max(2, len(sampled) * 6 // 60 + 2)]}
    write_json(lesson.root / 'reviewed' / f'{key}.json', data)
    return data


def load(workspace, key, data):
    if data['hash'] != sha(encoded([data['examples']])):
        raise ValueError('Набор изменён после подготовки.')
    lesson = StyleLesson(workspace)
    history = {m['message_id']: m for m in lesson.store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    reviews = {item['case_id']: item for item in training_examples(workspace) + action_examples(workspace)}
    valid = {item['case_id']: item for item in holdout_examples(workspace, data.get('holdout_ref', 'holdout.json'))}
    ids = {}
    for split, items in data['examples'].items():
        if len(items) != data['counts'][split]:
            raise ValueError('Неполный набор.')
        ids[split] = set()
        for item in items:
            ids[split].update(item['ids'])
            if split == 'valid':
                good = valid.get(item['case_id']) == item
            elif item['case_id'].startswith(('history:', 'history_action:')):
                source_ids = item['context_ids'] + item['response_ids']
                if source_ids != item['ids'] or any(i not in history or i in deleted for i in source_ids):
                    raise ValueError('Источник обучения удалён или отсутствует.')
                source = [history[i] for i in source_ids]
                size = len(item['context_ids'])
                expected_row = (training_row(source[:size], 'reply')
                                if item['case_id'].startswith('history_action:')
                                else example(source[:size], source[size:]))
                good = (all(m['sender'] == 'Я' for m in source[size:])
                        and digest(source) == item['source_digest']
                        and expected_row == item['row'])
            else:
                good = reviews.get(item['case_id']) == item
            if not good:
                raise ValueError('История или оценки изменились. Подготовьте набор заново.')
    reserved = {i for key in lesson.holdout_for(data.get('holdout_ref', 'holdout.json'))['cases']
                for case in [lesson.store.get(key)] for i in case['context_ids'] + case['expected_ids']}
    if not ids['train'] or not ids['valid'] or max(ids['train']) >= min(reserved):
        raise ValueError('Учебные источники пересекают проверочную границу.')
    from mybot.services.reply_evaluation import reserved as evaluation_reserved
    if ids['train'] & evaluation_reserved(workspace):
        raise ValueError('Учебные источники пересекаются с проверкой готовых ответов.')
    return data, {split: [item['row'] for item in items] for split, items in data['examples'].items()}


def prepare_repair(workspace, tokenizer, model_path, version):
    """New manifest, same sources; never re-sample or overwrite the old dataset."""
    from mybot.services.style_versions import adapter
    from mybot.services.reviewed_dataset import load_reviewed
    root = adapter(workspace, version, require_quality=False)
    run = read_json(root / 'run.json')
    manifest, rows = load_reviewed(workspace, run['dataset_id'])
    if manifest.get('version') != 2 or manifest.get('purpose') != 'history':
        raise ValueError('Восстановление поддерживает первоначальный набор истории.')
    audit = StyleLesson(workspace).holdout_audit(manifest.get('holdout_ref', 'holdout.json'))
    if any(item['reason'] for item in audit):
        raise ValueError('Старый набор проверки содержит переход через паузу. '
                         'Подготовьте исправленную проверку и новое обучение на истории.')
    history = {m['message_id']: m for m in StyleLesson(workspace).store.history()}
    threshold = gap_seconds(workspace)
    for item in manifest['examples']['train']:
        if item['case_id'].startswith(('history:', 'history_action:')):
            context = [history[i] for i in item['context_ids']]
            expected = [history[i] for i in item['response_ids']]
            if crossing_reason(context, expected, threshold):
                raise ValueError('Старый набор обучения содержит ответ после паузы. '
                                 'Подготовьте новое обучение на истории.')
    for items in rows.values():
        for row in items:
            tokens, _ = training_tokens(tokenizer, row)
            if len(tokens) > manifest['max_length']:
                raise ValueError('Исходный пример не помещается в новый формат; набор сохранён без изменений.')
    manifest.update(id=uuid4().hex, format_version=FORMAT_VERSION, created_at=now(),
                    learning_rate=2e-5, parent_adapter=None, repair_of=version,
                    tokenizer_hash=tokenizer_hash(model_path))
    write_json(workspace.root / 'learning/reviewed' / (manifest['id'] + '.json'), manifest)
    return manifest

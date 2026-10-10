"""Validated owner-selected style examples, with source-disjoint evaluation."""

import re
from uuid import uuid4

from mybot.services.learning_dataset import encoded, example, sha
from mybot.services.practice_learning import chosen_reply
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, now
from mybot.storage.style_lesson import StyleLesson
from mybot.ai.local_format import FORMAT_VERSION
from mybot.services.decision_task import training_row
from mybot.services.conversation_boundaries import crossing_reason, gap_seconds

MIN_TRAIN = 8
MIN_VALID = 2


def training_examples(workspace):
    items = reviewed_examples(workspace)
    reserved = StyleLesson(workspace).reserved_ids()
    return [item for item in items if max(item['ids']) < min(reserved)] if reserved else items


def reviewed_examples(workspace):
    store = PracticeStore(workspace)
    messages = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    result = []
    for case in store.records():
        try:
            context, expected = store.source(case, messages=messages, deleted=deleted)
        except ValueError:
            continue
        if crossing_reason(context, expected, gap_seconds(workspace)):
            continue
        chosen = chosen_reply(case, expected)
        if chosen is None:
            continue
        kind, text = chosen
        row = example(context, [{'text': text}])
        result.append({'case_id': case['id'], 'source_kind': kind,
                       'source_digest': case['source_digest'],
                       'ids': case['context_ids'] + case['expected_ids'], 'row': row})
    latest = {}
    for path in (workspace.root / 'learning/confirmed').glob('*.json'):
        evidence = read_json(path)
        if (evidence.get('account_id'), evidence.get('dialog_id')) != (workspace.account_id, workspace.dialog_id):
            raise ValueError('Подтверждённый ответ принадлежит другому диалогу.')
        ids = tuple(evidence['response_ids'])
        if ids not in latest or evidence['confirmed_at'] > latest[ids][1]['confirmed_at']:
            latest[ids] = (path.stem, evidence)
    replaced = set()
    for review_id, evidence in latest.values():
        case = store.get(evidence['case_id'])
        try:
            context, expected = store.source(case, messages=messages, deleted=deleted)
        except ValueError:
            continue
        if crossing_reason(context, expected, gap_seconds(workspace)):
            continue
        replaced.add(case['id'])
        result.append({'case_id': case['id'], 'source_kind': 'owner_confirmed_reply',
                       'review_id': review_id, 'source_digest': case['source_digest'],
                       'ids': case['context_ids'] + case['expected_ids'],
                       'row': example(context, [{'text': evidence['text']}])})
    result = [item for item in result if item.get('review_id') or item['case_id'] not in replaced]
    return sorted(result, key=lambda item: max(item['ids']))


def action_examples(workspace):
    """Only owner-confirmed actions; an unanswered export is never a WAIT label."""
    store = PracticeStore(workspace)
    messages = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    result = []
    for case in store.records():
        action = (case.get('owner_action') or {}).get('action')
        if action not in {'reply', 'wait'} or case.get('purpose') == 'holdout':
            continue
        try:
            context, _ = store.source(case, messages=messages, deleted=deleted)
        except ValueError:
            continue
        if case['expected_ids']:
            _, expected = store.source(case, messages=messages, deleted=deleted)
            if crossing_reason(context, expected, gap_seconds(workspace)):
                continue
        result.append({'case_id': 'action:' + case['id'], 'source_kind': 'owner_confirmed_action',
                       'source_digest': case['source_digest'],
                       'ids': case['context_ids'] + case['expected_ids'],
                       'row': training_row(context, action)})
    return sorted(result, key=lambda item: max(item['ids']))


def split_examples(items):
    if len(items) < MIN_TRAIN + MIN_VALID:
        raise ValueError(f'Выбрано {len(items)} ответов. Нужно минимум {MIN_TRAIN + MIN_VALID}; '
                         'это только небольшой пробный запуск, не гарантия качества.')
    valid = items[-max(MIN_VALID, len(items) // 5):]
    cutoff = min(min(item['ids']) for item in valid)
    train = [item for item in items if max(item['ids']) < cutoff]
    # Deduplicate exact examples, including repeated answers in the holdout.
    seen = {sha(encoded([item['row']])) for item in valid}
    unique = []
    for item in train:
        fingerprint = sha(encoded([item['row']]))
        if fingerprint not in seen:
            unique.append(item)
            seen.add(fingerprint)
    if len(unique) < MIN_TRAIN:
        raise ValueError(f'После отделения проверки осталось {len(unique)} обучающих примеров '
                         f'(нужно {MIN_TRAIN}). Оцените ещё отрывки из разных частей истории.')
    return {'train': unique, 'valid': valid}


def prepare_reviewed(workspace):
    lesson = StyleLesson(workspace)
    if lesson.holdout():
        reserved = lesson.reserved_ids()
        train = training_examples(workspace)
        if len(train) < MIN_TRAIN:
            raise ValueError(f'Пригодных выбранных ответов: {len(train)}. Нужно минимум {MIN_TRAIN}. Продолжите занятие.')
        splits = {'train': train, 'valid': holdout_examples(workspace)}
    else:
        splits = split_examples(reviewed_examples(workspace))
    key = uuid4().hex
    manifest = {'version': 1, 'id': key, 'account_id': workspace.account_id,
                'format_version': FORMAT_VERSION,
                'holdout_ref': lesson.holdout_ref(),
                'dialog_id': workspace.dialog_id, 'created_at': now(),
                'counts': {name: len(rows) for name, rows in splits.items()},
                'examples': splits, 'hash': sha(encoded([splits]))}
    manifest['fixed_holdout'] = bool(lesson.holdout())
    write_json(workspace.root / 'learning' / 'reviewed' / f'{key}.json', manifest)
    return manifest


def load_reviewed(workspace, key):
    if not re.fullmatch(r'[a-f0-9]{32}', key):
        raise ValueError('Некорректный ID набора.')
    data = read_json(workspace.root / 'learning' / 'reviewed' / f'{key}.json')
    if (not data or data.get('version') not in {1, 2, 3, 4} or data.get('id') != key
            or data.get('account_id') != workspace.account_id
            or data.get('dialog_id') != workspace.dialog_id):
        raise ValueError('Набор недоступен в этом диалоге.')
    if data['version'] == 4:
        from mybot.services.online_dataset import load
        return load(workspace, data)
    if data['version'] == 3:
        from mybot.services.full_history_dataset import load_full
        return load_full(workspace, data)
    if data['version'] == 2:
        from mybot.services.history_training import load
        return load(workspace, key, data)
    if data['hash'] != sha(encoded([data['examples']])):
        raise ValueError('Набор изменён после подготовки.')
    current = {item['case_id']: item for item in reviewed_examples(workspace)}
    if data.get('fixed_holdout'):
        current.update({item['case_id']: item for item in holdout_examples(workspace,
                        data.get('holdout_ref', 'holdout.json'))})
    ids = {}
    for split, items in data['examples'].items():
        if len(items) != data['counts'][split]:
            raise ValueError('Набор неполон.')
        ids[split] = set()
        for item in items:
            if current.get(item['case_id']) != item:
                raise ValueError('Оценки или исходная история изменились. Подготовьте набор заново.')
            ids[split].update(item['ids'])
    if ids['train'] & ids['valid']:
        raise ValueError('Обучение пересекается с проверкой.')
    from mybot.services.reply_evaluation import reserved as evaluation_reserved
    if ids['train'] & evaluation_reserved(workspace):
        raise ValueError('Учебные источники пересекаются с проверкой готовых ответов.')
    if data.get('fixed_holdout') and max(ids['train']) >= min(ids['valid']):
        raise ValueError('Обучение содержит будущую для проверки историю.')
    return data, {split: [item['row'] for item in items]
                  for split, items in data['examples'].items()}


def holdout_examples(workspace, holdout_ref=None):
    lesson = StyleLesson(workspace)
    data = lesson.holdout_for(holdout_ref) if holdout_ref else lesson.holdout()
    result = []
    for key in (data or {}).get('cases', []):
        case = lesson.store.get(key)
        context, expected = lesson.store.source(case)
        result.append({'case_id': key, 'source_kind': 'heldout_historical_reply',
                       'source_digest': case['source_digest'],
                       'ids': case['context_ids'] + case['expected_ids'], 'row': example(context, expected)})
    return result

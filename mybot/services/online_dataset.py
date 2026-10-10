"""Immutable incremental datasets; frozen validation sources never become replay."""

import random
from uuid import uuid4

from mybot.ai.local_format import FORMAT_VERSION
from mybot.services.full_history_dataset import fit, seal
from mybot.services.history_training import tokenizer_hash
from mybot.services.learning_dataset import example
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, digest
from mybot.storage.style_lesson import StyleLesson


def reserved_ids(workspace):
    lesson = StyleLesson(workspace)
    from mybot.services.reply_evaluation import reserved as evaluation_reserved
    reserved = evaluation_reserved(workspace)
    for path in lesson.root.glob('holdout*.json'):
        for key in read_json(path, {}).get('cases', []):
            case = lesson.store.get(key)
            reserved.update(case['context_ids'] + case['expected_ids'])
    for path in (lesson.root / 'reviewed').glob('*.json'):
        for item in read_json(path, {}).get('examples', {}).get('valid', []):
            reserved.update(item['ids'])
    return reserved


def prepare(workspace, pending, parent, tokenizer, model_path):
    from mybot.services.reviewed_dataset import load_reviewed
    from mybot.services.style_versions import adapter
    root = adapter(workspace, parent)
    parent_run = read_json(root / 'run.json')
    original, _ = load_reviewed(workspace, parent_run['dataset_id'])
    replay_manifest = original
    ancestor = parent_run
    seen = {parent}
    while ancestor.get('purpose') == 'online':
        key = ancestor['parent_adapter']
        if key in seen:
            raise ValueError('Циклическая цепочка адаптеров.')
        seen.add(key)
        ancestor = read_json(workspace.root / 'learning/mlx_runs' / key / 'run.json')
        replay_manifest, _ = load_reviewed(workspace, ancestor['dataset_id'])
    if parent_run.get('lora_config') and parent_run['lora_config'] != __import__(
            'mybot.services.full_local_training', fromlist=['LORA_CONFIG']).LORA_CONFIG:
        raise ValueError('Конфигурация исходного LoRA несовместима.')
    messages = {m['message_id']: m for m in PracticeStore(workspace).history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    reserved = reserved_ids(workspace)
    maximum = original.get('max_length', 2048)
    train, excluded = [], []
    for item in pending:
        ids = item['context_ids'] + item['response_ids']
        reason = None
        if set(ids) & (reserved | deleted) or any(i not in messages for i in ids):
            reason = 'deleted_or_reserved_source'
        else:
            context = [messages[i] for i in item['context_ids']]
            response = [messages[i] for i in item['response_ids']]
            if digest(context + response) != item['source_digest']:
                reason = 'source_changed'
            else:
                target = response
                if item.get('source_kind') == 'owner_confirmed_reply':
                    from mybot.services.simple_lessons import confirmed
                    evidence = confirmed(workspace, item)
                    target = [{'text': evidence['text']}]
                fitted = fit(context, target, tokenizer, maximum)
                if fitted:
                    context, row = fitted
                    train.append({'case_id': 'online:' + item['id'], 'event_id': item['id'],
                        'context_ids': [m['message_id'] for m in context], 'response_ids': item['response_ids'],
                        'ids': [m['message_id'] for m in context + response],
                        'source_digest': digest(context + response), 'row': row,
                        'source_kind': item.get('source_kind', 'actual_sent_reply'),
                        **({k: item[k] for k in ('review_id', 'text')} | {'original_source_digest': item['source_digest']}
                           if item.get('source_kind') == 'owner_confirmed_reply' else {})})
                else:
                    reason = 'full_target_does_not_fit'
        if reason:
            excluded.append({'event_id': item['id'], 'reason': reason})
    if len(train) < 10:
        return None, excluded
    corrected_ids = {i for path in (workspace.root / 'learning/confirmed').glob('*.json')
                     for i in read_json(path, {}).get('response_ids', [])}
    corrected_ids.update(i for case in PracticeStore(workspace).records()
                         if case.get('owner_action', {}).get('action') == 'wait'
                         for i in case['expected_ids'])
    replay = [v for v in replay_manifest['examples']['train']
              if not set(v['ids']) & (reserved | deleted | corrected_ids)
              and all(i in messages for i in v['ids'])]
    cycle_id = uuid4().hex
    replay = random.Random(int(cycle_id[:8], 16)).sample(replay, min(40, len(replay)))
    # Preserve reviewed targets without replacing them with raw owner messages.
    for item in replay:
        item = dict(item, source_digest=digest([messages[i] for i in item['ids']]), replay=True)
        item.pop('event_id', None)
        train.append(item)
    data = {'version': 4, 'id': cycle_id, 'purpose': 'online', 'mode': 'online',
            'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
            'parent_adapter': parent, 'parent_sha256': parent_run['adapter_sha256'],
            'format_version': FORMAT_VERSION, 'tokenizer_hash': tokenizer_hash(model_path),
            'max_length': maximum, 'holdout_ref': original['holdout_ref'],
            'examples': {'train': train, 'valid': original['examples']['valid']},
            'counts': {'train': len(train), 'valid': len(original['examples']['valid'])},
            'order': list(range(len(train))), 'base_steps': len(train),
            'coverage_counts': {'new': len(train) - len(replay), 'replay': len(replay)},
            'new_events': [v['event_id'] for v in train if 'event_id' in v], 'excluded': excluded}
    random.Random(42).shuffle(data['order'])
    data['hash'] = seal(data)
    write_json(workspace.root / 'learning/reviewed' / (data['id'] + '.json'), data)
    load(workspace, data)
    return data, excluded


def load(workspace, data):
    if data.get('hash') != seal(data):
        raise ValueError('Набор изменился.')
    messages = {m['message_id']: m for m in PracticeStore(workspace).history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    ids = {}
    for split, items in data['examples'].items():
        ids[split] = set()
        if len(items) != data['counts'][split]:
            raise ValueError('Неполный набор.')
        for item in items:
            source_ids = item['ids']
            ids[split].update(source_ids)
            if set(source_ids) & deleted or any(i not in messages for i in source_ids):
                raise ValueError('Источники удалены.')
            if digest([messages[i] for i in source_ids]) != item['source_digest']:
                raise ValueError('Исходная история изменилась.')
            if item.get('source_kind') == 'owner_confirmed_reply':
                from mybot.services.simple_lessons import confirmed
                evidence = confirmed(workspace, dict(item, id=item['event_id']))
                if example([messages[i] for i in item['context_ids']], [{'text': evidence['text']}]) != item['row']:
                    raise ValueError('Подтверждённый ответ изменился.')
            elif item.get('event_id') and example([messages[i] for i in item['context_ids']],
                    [messages[i] for i in item['response_ids']]) != item['row']:
                raise ValueError('Фактический ответ изменился.')
    if ids['train'] & (ids['valid'] | reserved_ids(workspace)):
        raise ValueError('Учебные источники пересекаются с проверкой.')
    if sorted(data['order']) != list(range(data['counts']['train'])):
        raise ValueError('Порядок должен содержать один полный проход.')
    return data, {split: [item['row'] for item in items] for split, items in data['examples'].items()}

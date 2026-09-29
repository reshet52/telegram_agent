"""Versioned, per-dialog SFT data with a chronological, purged holdout."""

import hashlib
import json
import re
from uuid import uuid4

from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, candidates, digest, now


SYSTEM = ('Продолжи личную переписку от лица владельца аккаунта в его манере. '
          'Учитывай только предшествующий контекст. Верни один естественный ответ '
          'без объяснений; несколько сообщений разделяй пустой строкой.')


def example(context, response):
    prompt = [{'role': 'system', 'content': SYSTEM}]
    for message in context:
        role = 'assistant' if message['sender'] == 'Я' else 'user'
        if prompt[-1]['role'] == role:
            prompt[-1]['content'] += '\n\n' + message['text']
        else:
            prompt.append({'role': role, 'content': message['text']})
    return {'prompt': prompt, 'completion': [
        {'role': 'assistant', 'content': '\n\n'.join(m['text'] for m in response)}]}


def encoded(rows):
    return ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare_dataset(workspace, limit=2000):
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('Размер набора должен быть от 1 до 10000 примеров.')
    store = PracticeStore(workspace)
    messages = store.history()
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    options = candidates(messages, deleted, set())
    if len(options) < 2:
        raise ValueError('Недостаточно текстовых ситуаций для обучения и отдельной проверки.')
    # The last fifth of source messages is held out. Drop every crossing window.
    cutoff = messages[max(1, int(len(messages) * .8))]['message_id']
    reserved = set()
    for case in store.records():
        # Keep practice source IDs reserved even when its source was later deleted.
        reserved.update(case['context_ids'] + case['expected_ids'])
    rows = {'train': [], 'valid': []}
    sources = {'train': [], 'valid': []}
    seen = set()
    skipped = 0
    for context, response in options:
        ids = [m['message_id'] for m in context + response]
        if max(ids) < cutoff and not reserved.intersection(ids):
            split = 'train'
        elif min(ids) >= cutoff:
            split = 'valid'
        else:
            skipped += 1
            continue
        row = example(context, response)
        fingerprint = sha(encoded([row]))
        if fingerprint in seen or len(rows[split]) >= limit:
            skipped += 1
            continue
        seen.add(fingerprint)
        rows[split].append(row)
        sources[split].append({'context_ids': [m['message_id'] for m in context],
                               'response_ids': [m['message_id'] for m in response],
                               'digest': digest(context + response)})
    if not rows['train'] or not rows['valid']:
        raise ValueError('После отделения проверочной истории и тренировочных отрывков '
                         'не осталось двух непересекающихся выборок. Нужна более длинная история.')
    key = uuid4().hex
    root = workspace.root / 'learning' / 'datasets' / key
    # A unique directory and manifest written last make interrupted exports unusable.
    root.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split in rows:
        content = encoded(rows[split])
        write_text(root / f'{split}.jsonl', content)
        hashes[split] = sha(content)
    write_json(root / 'sources.json', sources)
    manifest = {'version': 1, 'id': key, 'account_id': workspace.account_id,
                'dialog_id': workspace.dialog_id, 'created_at': now(),
                'cutoff_message_id': cutoff, 'reserved_practice_ids': sorted(reserved),
                'counts': {split: len(items) for split, items in rows.items()},
                'skipped': skipped, 'hashes': hashes,
                'sources_hash': sha(encoded([sources])), 'target': 'one_reply',
                'status': 'prepared'}
    write_json(root / 'manifest.json', manifest)
    return manifest


def load_dataset(workspace, key):
    if not re.fullmatch(r'[a-f0-9]{32}', key):
        raise ValueError('Некорректный ID набора.')
    root = workspace.root / 'learning' / 'datasets' / key
    manifest = read_json(root / 'manifest.json')
    if (not isinstance(manifest, dict) or manifest.get('version') != 1
            or manifest.get('account_id') != workspace.account_id
            or manifest.get('dialog_id') != workspace.dialog_id or manifest.get('id') != key):
        raise ValueError('Набор не завершён или относится к другому диалогу.')
    sources = read_json(root / 'sources.json')
    if sha(encoded([sources])) != manifest['sources_hash']:
        raise ValueError('Изменены ссылки на источники набора.')
    store = PracticeStore(workspace)
    history = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    reserved = set()
    for case in store.records():
        reserved.update(case['context_ids'] + case['expected_ids'])
    split_ids = {}
    rows = {}
    for split in ('train', 'valid'):
        content = (root / f'{split}.jsonl').read_text(encoding='utf-8')
        if sha(content) != manifest['hashes'][split]:
            raise ValueError('Файл примеров изменился после подготовки.')
        rows[split] = [json.loads(line) for line in content.splitlines()]
        if len(rows[split]) != manifest['counts'][split] or len(rows[split]) != len(sources[split]):
            raise ValueError('Неполный набор примеров.')
        split_ids[split] = set()
        for source, row in zip(sources[split], rows[split]):
            ids = source['context_ids'] + source['response_ids']
            if any(key not in history or key in deleted for key in ids):
                raise ValueError('Источник удалён; подготовьте новый набор.')
            messages = [history[key] for key in ids]
            count = len(source['context_ids'])
            if digest(messages) != source['digest'] or example(messages[:count], messages[count:]) != row:
                raise ValueError('История изменилась; подготовьте новый набор.')
            split_ids[split].update(ids)
    if split_ids['train'] & (split_ids['valid'] | reserved):
        raise ValueError('Обучение пересекается с проверкой; подготовьте новый набор.')
    return manifest, rows

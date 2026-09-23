"""Idempotent two-file episode publication with a durable pending batch."""

import json
from pathlib import Path
from mybot.storage.atomic import read_json, write_json, write_text


def pending_path(episodes_filename):
    return Path(episodes_filename).with_name('episode_pending_batch.json')


def append_missing(path, records):
    path = Path(path)
    existing = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
                if line.strip()] if path.exists() else []
    known = {item['episode_id'] for item in existing}
    for item in records:
        if item['episode_id'] not in known:
            existing.append(item)
            known.add(item['episode_id'])
    write_text(path, ''.join(json.dumps(item, ensure_ascii=False) + '\n' for item in existing))


def recover_batch(episodes_filename, embeddings_filename):
    path = pending_path(episodes_filename)
    pending = read_json(path)
    if not pending:
        return
    if pending.get('embeddings_name') != Path(embeddings_filename).name:
        raise RuntimeError('Файл embeddings не соответствует checkpoint эпизодов.')
    append_missing(embeddings_filename, pending['embeddings'])
    append_missing(episodes_filename, pending['episodes'])
    write_json(path, None)


def commit_batch(episodes, embeddings, episodes_filename, embeddings_filename):
    if len(episodes) != len(embeddings):
        raise ValueError('Количество embeddings не соответствует эпизодам.')
    recover_batch(episodes_filename, embeddings_filename)
    write_json(pending_path(episodes_filename), {
        'episodes': episodes, 'embeddings': embeddings,
        'embeddings_name': Path(embeddings_filename).name,
    })
    recover_batch(episodes_filename, embeddings_filename)

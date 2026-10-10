"""Explicit, bounded repair of a saved history gap inside a staged transaction."""

import json
import shutil
from uuid import uuid4

from mybot.episodes.incremental import (
    IncrementalEpisodeTracker, PREVIOUS_CONTEXT_LIMIT, create_embeddings,
)
from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.deletions import load_deleted_message_ids


async def repair_episodes(transaction, messages, first_missing_id, progress):
    staged = transaction.staged
    old = [json.loads(line) for line in staged.episodes.read_text().splitlines() if line.strip()]
    # Reopen the immediately preceding episode too: a missed outgoing message
    # may extend its response rather than belong to the next episode.
    affected = next((i for i, e in enumerate(old)
                     if max(m['message_id'] for m in e['incoming'] + e['response']) >= first_missing_id), len(old))
    keep = old[:max(0, affected - 1)]
    checkpoint = keep[-1]['response'][-1]['message_id'] if keep else 0
    deleted = load_deleted_message_ids(staged.deleted_message_ids)
    active = [m for m in messages if m['message_id'] not in deleted]
    start = next((i + 1 for i, m in enumerate(active) if m['message_id'] == checkpoint), 0)
    if checkpoint and not start:
        raise ValueError('Контрольная точка восстановления отсутствует в истории.')
    tracker = IncrementalEpisodeTracker(keep[-1]['episode_id'] + 1 if keep else 1,
                                       active[max(0, start - PREVIOUS_CONTEXT_LIMIT):start])
    rebuilt = []
    for message in active[start:]:
        episode = tracker.process_message(message)
        if episode:
            rebuilt.append(episode)
    if len(rebuilt) > 256:
        raise ValueError(f'Восстановление затрагивает {len(rebuilt)} эпизодов. '
                         'Нужен отдельный запуск большого восстановления; данные не изменены.')
    await progress(f'Восстановление: пересобираю {len(rebuilt)} эпизодов после пропуска…')
    embeddings = await create_embeddings(rebuilt)
    if len(embeddings) != len(rebuilt):
        raise ValueError('Не получены все embeddings восстановленных эпизодов.')
    keep_ids = {e['episode_id'] for e in keep}
    base = read_json(staged.episode_embeddings, {})
    if staged.episode_embeddings.exists():
        base['episodes'] = [e for e in base.get('episodes', []) if e['episode_id'] in keep_ids]
        write_json(staged.episode_embeddings, base)
    live = ([json.loads(line) for line in staged.episode_embeddings_live.read_text().splitlines() if line.strip()]
            if staged.episode_embeddings_live.exists() else [])
    write_text(staged.episode_embeddings_live, ''.join(json.dumps(e) + '\n' for e in
               [e for e in live if e['episode_id'] in keep_ids] + embeddings))
    write_text(staged.episodes, ''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in keep + rebuilt))
    # Retain the pre-repair files after the transaction's temporary directory closes.
    backup = transaction.workspace.root / 'history_recovery_backups' / uuid4().hex
    backup.parent.mkdir(exist_ok=True)
    shutil.copytree(transaction.directory / 'before', backup)
    write_json(backup / 'repair.json', {'first_missing_id': first_missing_id,
                                     'preserved_episodes': len(keep), 'rebuilt_episodes': len(rebuilt)})
    return tracker, rebuilt

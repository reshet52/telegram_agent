"""Repair missing episode embeddings without rebuilding or renumbering episodes."""

import json
from mybot.episodes.incremental import create_embeddings
from mybot.episodes.checkpoint import commit_batch
from mybot.storage.atomic import durable_call, read_json


async def ensure_episode_embeddings(workspace, progress):
    if not workspace.episodes.exists():
        return
    episodes = [json.loads(line) for line in workspace.episodes.read_text(encoding='utf-8').splitlines()
                if line.strip()]
    base = read_json(workspace.episode_embeddings, {})
    indexed = {item['episode_id'] for item in base.get('episodes', []) if item.get('embedding')}
    if workspace.episode_embeddings_live.exists():
        for line in workspace.episode_embeddings_live.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get('embedding'):
                    indexed.add(item['episode_id'])
    missing = [item for item in episodes if item['episode_id'] not in indexed]
    for start in range(0, len(missing), 32):
        batch = missing[start:start + 32]
        embeddings = await create_embeddings(batch, batch_size=32)
        await durable_call(commit_batch, batch, embeddings, workspace.episodes, workspace.episode_embeddings_live)
        await progress(f'Индекс эпизодов: восстановлено {min(start + 32, len(missing))}/{len(missing)}')

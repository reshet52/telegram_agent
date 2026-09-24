"""Fold live indexes and memories into one dialog's base files without API calls."""

import json
from pathlib import Path

from mybot.config import Config
from mybot.episodes.checkpoint import recover_batch
from mybot.storage.atomic import read_json, write_json, write_text


def _jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path.name}:{number}: ожидался JSON-объект")
                records.append(record)
    return records


def _memory_key(record):
    category, memory = record.get("category"), record.get("memory")
    if not isinstance(category, str) or not isinstance(memory, str) or not memory.strip():
        raise ValueError("Запись памяти без категории или текста")
    return category, memory.strip()


def _compact_episodes(workspace):
    live = _jsonl(workspace.episode_embeddings_live)
    if not live:
        return 0
    base = read_json(workspace.episode_embeddings,
        {"model": Config.EMBEDDING_MODEL, "episodes": []})
    if not isinstance(base, dict) or not isinstance(base.get("episodes"), list):
        raise ValueError("Некорректный основной индекс эпизодов")
    merged = {}
    for item in [*base["episodes"], *live]:
        episode_id = item.get("episode_id") if isinstance(item, dict) else None
        if not isinstance(episode_id, int) or not item.get("embedding"):
            raise ValueError("Некорректная запись embedding эпизода")
        merged[episode_id] = item  # The live version takes precedence.
    base["episodes"] = list(merged.values())
    base["episode_count"] = len(merged)
    write_json(workspace.episode_embeddings, base)
    # If interrupted before this line, retrying deduplicates the same IDs.
    write_text(workspace.episode_embeddings_live, "")
    return len(live)


def _compact_memories(workspace):
    live_memory = _jsonl(workspace.agent_memory_live)
    live_embeddings = _jsonl(workspace.memory_embeddings_live)
    if not live_memory and not live_embeddings:
        return 0
    base_memory = read_json(workspace.agent_memory, {})
    if not isinstance(base_memory, dict) or any(
            not isinstance(items, list) for items in base_memory.values()):
        raise ValueError("Некорректный основной файл памяти")
    base_index = read_json(workspace.memory_embeddings,
        {"model": Config.EMBEDDING_MODEL, "memories": []})
    if isinstance(base_index, list):
        items, index_key = base_index, None
    elif isinstance(base_index, dict):
        index_key = "memories" if "memories" in base_index else "items"
        items = base_index.get(index_key, [])
    else:
        raise ValueError("Некорректный основной индекс памяти")
    if not isinstance(items, list):
        raise ValueError("Некорректный список embeddings памяти")
    indexed = {}
    for record in [*items, *live_embeddings]:
        if not isinstance(record, dict) or not record.get("embedding"):
            raise ValueError("Некорректная запись embedding памяти")
        indexed[_memory_key(record)] = record
    if any(_memory_key(record) not in indexed for record in live_memory):
        raise ValueError("Live-память не имеет соответствующего embedding; файлы сохранены")

    for record in live_memory:
        category, memory = _memory_key(record)
        group = base_memory.setdefault(category, [])
        if not isinstance(group, list):
            raise ValueError("Некорректная категория основной памяти")
        item = {key: value for key, value in record.items() if key != "category"}
        existing = next((i for i, old in enumerate(group)
                         if isinstance(old, dict) and old.get("memory", "").strip() == memory), None)
        if existing is None:
            group.append(item)
        else:
            group[existing] = {**group[existing], **item}
    if index_key is None:
        base_index = list(indexed.values())
    else:
        base_index[index_key] = list(indexed.values())
        base_index["count"] = len(indexed)
    write_json(workspace.agent_memory, base_memory)
    write_json(workspace.memory_embeddings, base_index)
    # Both base files are durable before either source is cleared.
    write_text(workspace.agent_memory_live, "")
    write_text(workspace.memory_embeddings_live, "")
    return len(live_memory)


def compact_workspace(workspace):
    """Idempotent consolidation for an inactive, isolated dialog workspace.

    chat_history.jsonl and episodes.jsonl already receive live messages directly.
    A failed merge leaves its live source intact for a later retry.
    """
    recover_batch(workspace.episodes, workspace.episode_embeddings_live)
    return {
        "episode_embeddings": _compact_episodes(workspace),
        "memories": _compact_memories(workspace),
    }

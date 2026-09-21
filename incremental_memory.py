import json
from pathlib import Path

from ai_client import (
    ai_client,
    analyze_incremental_memory
)

from config import Config
from mybot.storage.history import load_all_messages
from style_analyzer import (
    build_full_history_text
)


BASE_EPISODE_EMBEDDINGS = Path(
    "memory/episode_embeddings.json"
)

EPISODES_FILE = Path(
    "memory/episodes.jsonl"
)

LIVE_MEMORY_FILE = Path(
    "memory/agent_memory_live.jsonl"
)

LIVE_MEMORY_EMBEDDINGS_FILE = Path(
    "memory/memory_embeddings_live.jsonl"
)

STATE_FILE = Path(
    "memory/memory_update_state.json"
)


def get_initial_memory_message_id():
    """
    Определяем границу старой памяти.

    Основной episode_embeddings.json
    был построен одновременно со старой
    agent_memory, поэтому последний
    episode из основной базы показывает
    приблизительную границу старых данных.
    """

    if not BASE_EPISODE_EMBEDDINGS.exists():
        return 0

    with open(
        BASE_EPISODE_EMBEDDINGS,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    base_episodes = data.get(
        "episodes",
        []
    )

    if not base_episodes:
        return 0

    last_episode_id = max(
        item["episode_id"]
        for item in base_episodes
    )

    with open(
        EPISODES_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if not line.strip():
                continue

            episode = json.loads(
                line
            )

            if (
                episode.get("episode_id")
                != last_episode_id
            ):
                continue

            response = episode.get(
                "response",
                []
            )

            if response:
                return (
                    response[-1].get(
                        "message_id",
                        0
                    )
                    or 0
                )

            incoming = episode.get(
                "incoming",
                []
            )

            if incoming:
                return (
                    incoming[-1].get(
                        "message_id",
                        0
                    )
                    or 0
                )

    return 0


def load_memory_state():
    if STATE_FILE.exists():
        with open(
            STATE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    initial_message_id = (
        get_initial_memory_message_id()
    )

    state = {
        "last_processed_message_id":
            initial_message_id
    }

    save_memory_state(
        state
    )

    return state


def save_memory_state(state):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=4
        )


def get_new_messages(
    last_processed_message_id
):
    messages = load_all_messages(
        Config.CHAT_HISTORY_JSONL
    )

    return [
        message
        for message in messages
        if (
            message.get(
                "message_id",
                0
            )
            > last_processed_message_id
        )
    ]


def append_memories(memories):
    if not memories:
        return

    with open(
        LIVE_MEMORY_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        for memory in memories:
            file.write(
                json.dumps(
                    memory,
                    ensure_ascii=False
                )
            )

            file.write("\n")


async def create_memory_embeddings(
    memories
):
    if not memories:
        return []

    texts = [
        memory["memory"]
        for memory in memories
    ]

    response = (
        await ai_client.embeddings.create(
            model=
                Config.EMBEDDING_MODEL,
            input=texts
        )
    )

    results = []

    for memory, item in zip(
        memories,
        response.data
    ):
        results.append({
            "category":
                memory["category"],

            "memory":
                memory["memory"],

            "confidence":
                memory.get(
                    "confidence",
                    "medium"
                ),

            "evidence":
                memory.get(
                    "evidence",
                    "single"
                ),

            "time_context":
                memory.get(
                    "time_context",
                    "later"
                ),

            "embedding":
                item.embedding
        })

    return results


def append_memory_embeddings(
    records
):
    if not records:
        return

    with open(
        LIVE_MEMORY_EMBEDDINGS_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
            )

            file.write("\n")


async def update_incremental_memory():
    state = load_memory_state()

    last_processed_message_id = (
        state.get(
            "last_processed_message_id",
            0
        )
    )

    new_messages = get_new_messages(
        last_processed_message_id
    )

    if not new_messages:
        return {
            "messages": 0,
            "memories": 0
        }

    history_text = (
        build_full_history_text(
            new_messages
        )
    )

    result_text = (
        await analyze_incremental_memory(
            history_text
        )
    )

    try:
        memories = json.loads(
            result_text
        )

    except json.JSONDecodeError:
        raise ValueError(
            "ИИ вернул некорректный JSON "
            "при обновлении памяти."
        )

    if not isinstance(
        memories,
        list
    ):
        raise ValueError(
            "Обновление памяти должно "
            "вернуть JSON-массив."
        )

    if memories:
        append_memories(
            memories
        )

        embeddings = (
            await create_memory_embeddings(
                memories
            )
        )

        append_memory_embeddings(
            embeddings
        )

    newest_message_id = max(
        message.get(
            "message_id",
            0
        )
        for message in new_messages
    )

    state[
        "last_processed_message_id"
    ] = newest_message_id

    save_memory_state(
        state
    )

    return {
        "messages":
            len(new_messages),

        "memories":
            len(memories)
    }
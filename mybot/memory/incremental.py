import json
from pathlib import Path

from mybot.ai.client import (
    ai_client,
    analyze_incremental_memory
)

from mybot.config import Config
from mybot.storage.history import load_all_messages
from mybot.ai.style_analyzer import (
    build_full_history_text
)


DEFAULT_BASE_EPISODE_EMBEDDINGS = Path(
    "memory/episode_embeddings.json"
)

DEFAULT_EPISODES_FILE = Path(
    "memory/episodes.jsonl"
)

DEFAULT_LIVE_MEMORY_FILE = Path(
    "memory/agent_memory_live.jsonl"
)

DEFAULT_LIVE_MEMORY_EMBEDDINGS_FILE = Path(
    "memory/memory_embeddings_live.jsonl"
)

DEFAULT_STATE_FILE = Path(
    "memory/memory_update_state.json"
)


def get_initial_memory_message_id(
    base_episode_embeddings_filename=
        DEFAULT_BASE_EPISODE_EMBEDDINGS,
    episodes_filename=
        DEFAULT_EPISODES_FILE
):
    base_embeddings_path = Path(
        base_episode_embeddings_filename
    )

    episodes_path = Path(
        episodes_filename
    )

    if (
        not base_embeddings_path.exists()
        or not episodes_path.exists()
    ):
        return 0

    with open(
        base_embeddings_path,
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
        episodes_path,
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


def load_memory_state(
    state_filename=
        DEFAULT_STATE_FILE,
    base_episode_embeddings_filename=
        DEFAULT_BASE_EPISODE_EMBEDDINGS,
    episodes_filename=
        DEFAULT_EPISODES_FILE
):
    state_path = Path(
        state_filename
    )

    if state_path.exists():
        with open(
            state_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    initial_message_id = (
        get_initial_memory_message_id(
            base_episode_embeddings_filename=
                base_episode_embeddings_filename,
            episodes_filename=
                episodes_filename
        )
    )

    state = {
        "last_processed_message_id":
            initial_message_id
    }

    save_memory_state(
        state,
        state_filename
    )

    return state


def save_memory_state(
    state,
    filename=DEFAULT_STATE_FILE
):
    file_path = Path(
        filename
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
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
    last_processed_message_id,
    history_filename=
        Config.CHAT_HISTORY_JSONL,
    deleted_filename=None
):
    raw_messages = load_all_messages(
        filename=history_filename,
        include_deleted=True,
        deleted_filename=
            deleted_filename
    )

    if last_processed_message_id:
        message_exists = any(
            message.get("message_id")
            == last_processed_message_id
            for message in raw_messages
        )

        if not message_exists:
            raise RuntimeError(
                "Последний обработанный "
                "memory message_id не найден "
                "в истории: "
                f"{last_processed_message_id}. "
                "История и состояние памяти "
                "рассинхронизированы."
            )

    messages = load_all_messages(
        filename=history_filename,
        deleted_filename=
            deleted_filename
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


def append_memories(
    memories,
    filename=
        DEFAULT_LIVE_MEMORY_FILE
):
    if not memories:
        return

    file_path = Path(
        filename
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
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
    records,
    filename=
        DEFAULT_LIVE_MEMORY_EMBEDDINGS_FILE
):
    if not records:
        return

    file_path = Path(
        filename
    )

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
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


async def update_incremental_memory(
    history_filename=
        Config.CHAT_HISTORY_JSONL,
    deleted_filename=None,
    base_episode_embeddings_filename=
        DEFAULT_BASE_EPISODE_EMBEDDINGS,
    episodes_filename=
        DEFAULT_EPISODES_FILE,
    live_memory_filename=
        DEFAULT_LIVE_MEMORY_FILE,
    live_memory_embeddings_filename=
        DEFAULT_LIVE_MEMORY_EMBEDDINGS_FILE,
    state_filename=
        DEFAULT_STATE_FILE
):
    state = load_memory_state(
        state_filename=
            state_filename,
        base_episode_embeddings_filename=
            base_episode_embeddings_filename,
        episodes_filename=
            episodes_filename
    )

    last_processed_message_id = (
        state.get(
            "last_processed_message_id",
            0
        )
    )

    new_messages = get_new_messages(
        last_processed_message_id,
        history_filename=
            history_filename,
        deleted_filename=
            deleted_filename
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
        # Сначала создаём embeddings.
        # Если OpenAI вернёт ошибку,
        # на диск ничего ещё не записано.
        embeddings = (
            await create_memory_embeddings(
                memories
            )
        )

        append_memories(
            memories,
            live_memory_filename
        )

        append_memory_embeddings(
            embeddings,
            live_memory_embeddings_filename
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
        state,
        state_filename
    )

    return {
        "messages":
            len(new_messages),

        "memories":
            len(memories)
    }
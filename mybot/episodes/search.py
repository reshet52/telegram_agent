import asyncio
import json
import math
from pathlib import Path

from openai import AsyncOpenAI

from mybot.storage.deletions import (
    load_deleted_message_ids
)

from config import Config


EPISODES_FILE = Path(
    "memory/episodes.jsonl"
)

BASE_EMBEDDINGS_FILE = Path(
    "memory/episode_embeddings.json"
)

LIVE_EMBEDDINGS_FILE = Path(
    "memory/episode_embeddings_live.jsonl"
)

TOP_K = 5
MIN_SIMILARITY = 0.30
QUERY_CONTEXT_MESSAGES = 8


client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)


def load_episodes():
    episodes = {}

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

            episodes[
                episode["episode_id"]
            ] = episode

    return episodes


def load_base_embeddings():
    if not BASE_EMBEDDINGS_FILE.exists():
        return []

    with open(
        BASE_EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(
            file
        )

    return data.get(
        "episodes",
        []
    )


def load_live_embeddings():
    if not LIVE_EMBEDDINGS_FILE.exists():
        return []

    embeddings = []

    with open(
        LIVE_EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(
                line
            )

            embeddings.append(
                item
            )

    return embeddings


def load_embeddings():
    base_embeddings = (
        load_base_embeddings()
    )

    live_embeddings = (
        load_live_embeddings()
    )

    # episode_id используется как ключ.
    # Поэтому даже если когда-нибудь
    # один episode случайно окажется
    # и в старом, и в live файле,
    # дубликата в поиске не будет.
    merged = {}

    for item in base_embeddings:
        episode_id = item.get(
            "episode_id"
        )

        if episode_id is None:
            continue

        merged[
            episode_id
        ] = item

    for item in live_embeddings:
        episode_id = item.get(
            "episode_id"
        )

        if episode_id is None:
            continue

        merged[
            episode_id
        ] = item

    return {
        "episodes": list(
            merged.values()
        )
    }


def message_to_text(message):
    text = message.get(
        "text"
    )

    if text:
        return text

    message_type = message.get(
        "type",
        "unknown"
    )

    return f"[{message_type}]"


def get_message_ids(messages):
    return {
        message.get("message_id")
        for message in messages
        if (
            isinstance(message, dict)
            and message.get(
                "message_id"
            ) is not None
        )
    }


def episode_overlaps(
    episode,
    excluded_message_ids
):
    if not excluded_message_ids:
        return False

    for section_name in (
        "previous_context",
        "incoming",
        "response"
    ):
        for message in episode.get(
            section_name,
            []
        ):
            message_id = message.get(
                "message_id"
            )

            if (
                message_id
                in excluded_message_ids
            ):
                return True

    return False


def build_episode_query(
    messages,
    max_messages=QUERY_CONTEXT_MESSAGES
):
    recent_messages = messages[
        -max_messages:
    ]

    lines = [
        "Текущая ситуация в диалоге:"
    ]

    for message in recent_messages:

        # Нужно для ручного теста
        # episode_search.py
        if isinstance(
            message,
            str
        ):
            sender = "sender"
            text = message

        else:
            sender = message.get(
                "sender",
                "Неизвестно"
            )

            text = message_to_text(
                message
            )

        lines.append(
            f"{sender}: {text}"
        )

    return "\n".join(
        lines
    )


async def create_query_embedding(
    text
):
    response = (
        await client.embeddings.create(
            model=
                Config.EMBEDDING_MODEL,
            input=[text]
        )
    )

    return (
        response.data[
            0
        ].embedding
    )


def cosine_similarity(
    vector_a,
    vector_b
):
    dot_product = sum(
        a * b
        for a, b in zip(
            vector_a,
            vector_b
        )
    )

    magnitude_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    magnitude_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if (
        magnitude_a == 0
        or magnitude_b == 0
    ):
        return 0.0

    return (
        dot_product
        / (
            magnitude_a
            * magnitude_b
        )
    )


def search_episodes(
    episodes,
    embeddings_data,
    query_embedding,
    top_k=TOP_K,
    min_similarity=
        MIN_SIMILARITY,
    exclude_message_ids=None
):
    results = []

    for item in embeddings_data[
        "episodes"
    ]:
        episode_id = item[
            "episode_id"
        ]

        episode = episodes.get(
            episode_id
        )

        if episode is None:
            continue

        if episode_overlaps(
            episode,
            exclude_message_ids
        ):
            continue

        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        if (
            score
            < min_similarity
        ):
            continue

        results.append({
            "score": score,
            "episode": episode
        })

    results.sort(
        key=lambda item:
            item["score"],
        reverse=True
    )

    return results[
        :top_k
    ]


async def find_similar_episodes(
    messages,
    top_k=TOP_K,
    min_similarity=
        MIN_SIMILARITY
):
    query_text = (
        build_episode_query(
            messages
        )
    )

    if not query_text.strip():
        return []

    episodes = load_episodes()

    embeddings_data = (
        load_embeddings()
    )

    query_embedding = (
        await create_query_embedding(
            query_text
        )
    )

    excluded_message_ids = (
        get_message_ids(
            messages
        )
    )

    deleted_message_ids = (
        load_deleted_message_ids()
    )

    excluded_message_ids.update(
        deleted_message_ids
    )

    return search_episodes(
        episodes=episodes,
        embeddings_data=
            embeddings_data,
        query_embedding=
            query_embedding,
        top_k=top_k,
        min_similarity=
            min_similarity,
        exclude_message_ids=
            excluded_message_ids
    )


def print_result(result):
    episode = result[
        "episode"
    ]

    score = result[
        "score"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        f'EPISODE '
        f'{episode["episode_id"]} '
        f'| similarity: '
        f'{score:.4f}'
    )

    print(
        "\nINCOMING:"
    )

    for message in episode[
        "incoming"
    ]:
        print(
            f'{message.get("sender")}: '
            f'{message_to_text(message)}'
        )

    print(
        "\nREAL RESPONSE:"
    )

    for message in episode[
        "response"
    ]:
        print(
            f'{message.get("sender")}: '
            f'{message_to_text(message)}'
        )


async def main():
    base_embeddings = (
        load_base_embeddings()
    )

    live_embeddings = (
        load_live_embeddings()
    )

    combined_embeddings = (
        load_embeddings()
    )

    print(
        "\nБаза эпизодов:"
    )

    print(
        f"Старых embeddings: "
        f"{len(base_embeddings)}"
    )

    print(
        f"Live embeddings: "
        f"{len(live_embeddings)}"
    )

    print(
        f"Всего для поиска: "
        f'{len(combined_embeddings["episodes"])}'
    )

    query = input(
        "\nВведите сообщение собеседника:\n"
    ).strip()

    if not query:
        print(
            "Сообщение пустое."
        )

        return

    results = (
        await find_similar_episodes(
            [query]
        )
    )

    if not results:
        print(
            "\nПодходящих исторических "
            "эпизодов не найдено."
        )

        return

    print(
        f"\nНайдено эпизодов: "
        f"{len(results)}"
    )

    for result in results:
        print_result(
            result
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
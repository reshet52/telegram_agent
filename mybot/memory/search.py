import asyncio
import json
import math
from pathlib import Path

from openai import AsyncOpenAI

from mybot.config import Config
from mybot.storage.history import load_last_messages


DEFAULT_BASE_EMBEDDINGS_FILE = Path(
    "memory/memory_embeddings.json"
)

DEFAULT_LIVE_EMBEDDINGS_FILE = Path(
    "memory/memory_embeddings_live.jsonl"
)

TOP_K = 8
MIN_SIMILARITY = 0.35
QUERY_CONTEXT_MESSAGES = 8


client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)


def load_base_embeddings(
    filename=
        DEFAULT_BASE_EMBEDDINGS_FILE
):
    file_path = Path(
        filename
    )

    if not file_path.exists():
        return []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if (
            "items" in data
            and isinstance(
                data["items"],
                list
            )
        ):
            return data["items"]

        if (
            "memories" in data
            and isinstance(
                data["memories"],
                list
            )
        ):
            return data["memories"]

    return []


def load_live_embeddings(
    filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE
):
    file_path = Path(
        filename
    )

    if not file_path.exists():
        return []

    items = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if not line.strip():
                continue

            items.append(
                json.loads(line)
            )

    return items


def load_all_embeddings(
    base_filename=
        DEFAULT_BASE_EMBEDDINGS_FILE,
    live_filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE
):
    base_items = (
        load_base_embeddings(
            base_filename
        )
    )

    live_items = (
        load_live_embeddings(
            live_filename
        )
    )

    print("\nБаза памяти:")

    print(
        f"Старых embeddings: "
        f"{len(base_items)}"
    )

    print(
        f"Live embeddings: "
        f"{len(live_items)}"
    )

    print(
        f"Всего для поиска: "
        f"{len(base_items) + len(live_items)}"
    )

    return (
        base_items
        + live_items
    )


def message_to_text(message):
    text = message.get("text")

    if text:
        return text

    message_type = message.get(
        "type",
        "unknown"
    )

    return f"[{message_type}]"


def build_memory_query(
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

    return "\n".join(lines)


async def create_query_embedding(text):
    response = await client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=[text]
    )

    return response.data[0].embedding


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
        / (magnitude_a * magnitude_b)
    )


def deduplicate_results(results):
    unique = []
    seen = set()

    for item in results:
        memory_text = item["memory"].get(
            "memory",
            ""
        ).strip()

        key = (
            item["memory"].get(
                "category",
                ""
            ),
            memory_text
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def search_memories(
    embeddings_data,
    query_embedding,
    top_k=TOP_K,
    min_similarity=MIN_SIMILARITY
):
    results = []

    for item in embeddings_data:
        embedding = item.get("embedding")

        if not embedding:
            continue

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        if score < min_similarity:
            continue

        results.append({
            "score": score,
            "memory": item
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    results = deduplicate_results(
        results
    )

    return results[:top_k]


async def find_relevant_memories(
    messages,
    top_k=TOP_K,
    min_similarity=
        MIN_SIMILARITY,
    base_embeddings_filename=
        DEFAULT_BASE_EMBEDDINGS_FILE,
    live_embeddings_filename=
        DEFAULT_LIVE_EMBEDDINGS_FILE
):
    query_text = build_memory_query(
        messages
    )

    if not query_text.strip():
        return []

    embeddings_data = (
        load_all_embeddings(
            base_filename=
                base_embeddings_filename,
            live_filename=
                live_embeddings_filename
        )
    )

    if not embeddings_data:
        return []

    query_embedding = (
        await create_query_embedding(
            query_text
        )
    )

    return search_memories(
        embeddings_data=
            embeddings_data,
        query_embedding=
            query_embedding,
        top_k=top_k,
        min_similarity=
            min_similarity
    )


def print_memory_result(
    result,
    index
):
    memory = result["memory"]
    score = result["score"]

    print("\n" + "=" * 70)

    print(
        f"MEMORY {index} | "
        f"similarity: {score:.4f}"
    )

    print(
        f"Категория: "
        f'{memory.get("category", "-")}'
    )

    print(
        f"Time context: "
        f'{memory.get("time_context", "-")}'
    )

    print(
        f"Confidence: "
        f'{memory.get("confidence", "-")}'
    )

    print()
    print(
        memory.get(
            "memory",
            "[пусто]"
        )
    )


async def main():
    messages = load_last_messages(
        filename=Config.CHAT_HISTORY_JSONL,
        count=10
    )

    print("\nПоследние сообщения:\n")

    for message in messages:
        sender = message.get("sender")
        text = message_to_text(message)
        print(f"{sender}: {text}")

    query_text = build_memory_query(
        messages
    )

    print("\nТекст для semantic search:\n")
    print(query_text)

    results = await find_relevant_memories(
        messages
    )

    if not results:
        print(
            "\nПодходящих записей памяти "
            "не найдено."
        )
        return

    print(
        f"\nНайдено релевантных записей: "
        f"{len(results)}"
    )

    for index, result in enumerate(
        results,
        start=1
    ):
        print_memory_result(
            result,
            index
        )


if __name__ == "__main__":
    asyncio.run(main())
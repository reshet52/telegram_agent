import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from mybot.config import Config


MEMORY_FILE = Path(
    "memory/agent_memory.json"
)

OUTPUT_FILE = Path(
    "memory/memory_embeddings.json"
)

BATCH_SIZE = 100


client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)


def load_agent_memory():
    with open(
        MEMORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def flatten_memories(agent_memory):
    memories = []

    memory_id = 1

    for category, items in agent_memory.items():

        if not isinstance(items, list):
            continue

        for item in items:

            if not isinstance(item, dict):
                continue

            memory_text = item.get("memory")

            if not memory_text:
                continue

            memories.append({
                "memory_id": memory_id,
                "category": category,
                "memory": memory_text,
                "confidence": item.get(
                    "confidence"
                ),
                "evidence": item.get(
                    "evidence"
                ),
                "time_context": item.get(
                    "time_context"
                )
            })

            memory_id += 1

    return memories


async def create_embeddings(texts):
    response = await client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=texts
    )

    embeddings = [
        item.embedding
        for item in response.data
    ]

    tokens = (
        response.usage.total_tokens
        if response.usage
        else 0
    )

    return embeddings, tokens


async def main():
    agent_memory = load_agent_memory()

    memories = flatten_memories(
        agent_memory
    )

    print(
        f"Всего записей памяти: "
        f"{len(memories)}"
    )

    all_results = []
    total_tokens = 0

    total_batches = (
        len(memories)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for start in range(
        0,
        len(memories),
        BATCH_SIZE
    ):
        batch = memories[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        print(
            f"Обрабатываю batch "
            f"{batch_number}/"
            f"{total_batches}..."
        )

        texts = [
            (
                f'Категория: '
                f'{memory["category"]}\n'
                f'Память: '
                f'{memory["memory"]}'
            )
            for memory in batch
        ]

        embeddings, tokens = (
            await create_embeddings(
                texts
            )
        )

        total_tokens += tokens

        for memory, embedding in zip(
            batch,
            embeddings
        ):
            result = dict(memory)

            result["embedding"] = (
                embedding
            )

            all_results.append(
                result
            )

    output_data = {
        "model": Config.EMBEDDING_MODEL,
        "count": len(all_results),
        "memories": all_results
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False
        )

    print("\nГотово.")

    print(
        f"Создано embeddings: "
        f"{len(all_results)}"
    )

    print(
        f"Всего токенов: "
        f"{total_tokens}"
    )

    print(
        f"Файл сохранён: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    asyncio.run(main())
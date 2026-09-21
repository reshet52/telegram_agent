import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from config import Config


EPISODES_FILE = Path(
    "memory/episodes.jsonl"
)

OUTPUT_FILE = Path(
    "memory/episode_embeddings.json"
)

BATCH_SIZE = 100


client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)


def load_episodes():
    episodes = []

    with open(
        EPISODES_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if not line.strip():
                continue

            episodes.append(
                json.loads(line)
            )

    return episodes


def message_to_text(message):
    text = message.get("text")

    if text:
        return text

    message_type = message.get(
        "type",
        "unknown"
    )

    return f"[{message_type}]"


def build_search_text(episode):
    texts = []

    for message in episode["incoming"]:
        texts.append(
            message_to_text(message)
        )

    return "\n".join(texts)


async def create_embeddings(texts):
    response = await client.embeddings.create(
        model=Config.EMBEDDING_MODEL,
        input=texts
    )

    return (
        [item.embedding for item in response.data],
        response.usage.total_tokens
    )


async def main():
    episodes = load_episodes()

    print(
        f"Загружено эпизодов: "
        f"{len(episodes)}"
    )

    indexed_episodes = []
    total_tokens = 0

    for start in range(
        0,
        len(episodes),
        BATCH_SIZE
    ):
        batch = episodes[
            start:start + BATCH_SIZE
        ]

        texts = [
            build_search_text(episode)
            for episode in batch
        ]

        batch_number = (
            start // BATCH_SIZE + 1
        )

        total_batches = (
            len(episodes)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        print(
            f"Обрабатываю batch "
            f"{batch_number}/{total_batches}..."
        )

        embeddings, tokens = (
            await create_embeddings(texts)
        )

        total_tokens += tokens

        for episode, embedding in zip(
            batch,
            embeddings
        ):
            indexed_episodes.append({
                "episode_id":
                    episode["episode_id"],

                "embedding":
                    embedding
            })

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "model":
                    Config.EMBEDDING_MODEL,

                "episode_count":
                    len(indexed_episodes),

                "total_tokens":
                    total_tokens,

                "episodes":
                    indexed_episodes
            },
            file
        )

    print()
    print("Готово.")
    print(
        f"Создано embeddings: "
        f"{len(indexed_episodes)}"
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
import asyncio
from pathlib import Path

from ai_client import analyze_full_history
from config import Config
from history_reader import (
    load_all_messages,
    split_messages_into_chunks
)
from style_analyzer import build_full_history_text


CHUNK_SIZE = 500


async def main():
    print("Загружаю историю...")

    messages = load_all_messages(
        Config.CHAT_HISTORY_JSONL
    )

    print(f"Всего сообщений: {len(messages)}")

    chunks = split_messages_into_chunks(
        messages,
        chunk_size=CHUNK_SIZE
    )

    print(f"Частей для анализа: {len(chunks)}")

    results_folder = Path("memory/chunk_analysis")
    results_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, chunk in enumerate(chunks, start=1):
        result_file = (
            results_folder /
            f"chunk_{index:03d}.txt"
        )

        if result_file.exists():
            print(
                f"Часть {index}/{len(chunks)} "
                f"уже проанализирована — пропускаю."
            )
            continue

        print(
            f"\nАнализирую часть "
            f"{index}/{len(chunks)}..."
        )

        history_part = build_full_history_text(chunk)

        try:
            analysis = await analyze_full_history(
                history_part
            )
        except Exception as error:
            print(
                f"\nОшибка при анализе части {index}: "
                f"{error}"
            )
            print(
                "Анализ остановлен. "
                "Следующий запуск продолжит с этого места."
            )
            break

        with open(
            result_file,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(analysis)

        print(
            f"Сохранено: {result_file}"
        )


    print("\nАнализ всех частей завершён.")


asyncio.run(main())
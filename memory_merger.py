import json
from pathlib import Path
import asyncio
from ai_client import merge_memory_category


CHUNKS_FOLDER = Path("memory/chunk_analysis")


def load_chunk_analyses():
    analyses = []

    chunk_files = sorted(
        CHUNKS_FOLDER.glob("chunk_*.txt")
    )

    for chunk_file in chunk_files:
        # Старый первый анализ нам не нужен
        if chunk_file.name == "chunk_001_old.txt":
            continue

        with open(
            chunk_file,
            "r",
            encoding="utf-8"
        ) as file:
            content = file.read().strip()

        try:
            analysis = json.loads(content)
        except json.JSONDecodeError as error:
            print(
                f"Не удалось прочитать "
                f"{chunk_file.name}: {error}"
            )
            continue

        analyses.append({
            "chunk": chunk_file.name,
            "analysis": analysis
        })

    return analyses


def collect_memory_data(analyses):
    memory = {
        "style_patterns": [],
        "behavior_patterns": [],
        "user_facts": [],
        "person_facts": [],
        "relationship_facts": [],
        "important_events": []
    }

    for item in analyses:
        analysis = item["analysis"]
        chunk_name = item["chunk"]

        for category in memory:
            values = analysis.get(category, [])

            for value in values:
                memory[category].append({
                    "source_chunk": chunk_name,
                    **value
                })

    return memory


async def merge_all_categories(memory):
    final_memory = {}

    for category, items in memory.items():
        print(
            f"\nОбъединяю {category} "
            f"({len(items)} записей)..."
        )

        result = await merge_memory_category(
            category,
            items
        )

        try:
            final_memory[category] = json.loads(
                result
            )
        except json.JSONDecodeError as error:
            print(
                f"Ошибка JSON в категории "
                f"{category}: {error}"
            )
            return None

    return final_memory


async def main():
    analyses = load_chunk_analyses()

    print(
        f"Успешно загружено анализов: "
        f"{len(analyses)}"
    )

    memory = collect_memory_data(analyses)

    print("\nСобрано:")

    for category, values in memory.items():
        print(
            f"{category}: {len(values)}"
        )

    final_memory = await merge_all_categories(
        memory
    )

    if final_memory is None:
        print(
            "\nОбъединение остановлено из-за ошибки."
        )
        return

    output_file = Path(
        "memory/agent_memory.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            final_memory,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"\nГотово. Итоговая память сохранена:"
    )
    print(output_file)


if __name__ == "__main__":
    asyncio.run(main())
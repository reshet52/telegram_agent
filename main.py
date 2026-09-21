from ai_client import generate_answers
from config import Config
from context_builder import build_ai_request
from mybot.telegram.dialogs import choose_dialog
from mybot.telegram.exporter import export_dialog
from mybot.storage.history import load_last_messages, print_messages
from memory_manager import load_agent_memory
from mybot.storage.history import load_all_messages
from style_analyzer import build_style_analysis_request
from mybot.telegram.client import client


async def main():
    if Config.UPDATE_HISTORY:
        selected_dialog = await choose_dialog(client)

        await export_dialog(
            client,
            selected_dialog
        )

    messages = load_last_messages(
        filename=Config.CHAT_HISTORY_JSONL,
        count=10
    )

    print_messages(messages)

    user_instruction = input(
        "\nОпиши, как ты хочешь ответить "
        "(или нажми Enter без указания):\n"
    ).strip()

    if not user_instruction:
        user_instruction = (
            "Ответь естественно, полностью сохраняя мой стиль общения."
        )

    memory = load_agent_memory()

    ai_request = await build_ai_request(
        messages,
        user_instruction,
        memory
    )

    with open(
        Config.AI_REQUEST_PREVIEW,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(ai_request)

    print("\nЗапрос для ИИ подготовлен.")
    print(
        f"Он сохранён в файле "
        f"{Config.AI_REQUEST_PREVIEW}"
    )

    print("\nОтправляю запрос ИИ...")

    try:
        answers = await generate_answers(ai_request)
    except Exception as error:
        print(f"\nОшибка OpenAI API: {error}")
        return

    print("\nВарианты ответа:\n")
    print(answers)

    if Config.BUILD_FULL_HISTORY_ANALYSIS:
        all_messages = load_all_messages(
            Config.CHAT_HISTORY_JSONL
        )

        analysis_request = build_style_analysis_request(
            all_messages
        )

        with open(
            Config.FULL_HISTORY_ANALYSIS,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(analysis_request)

        print(
            "\nПолная история подготовлена для анализа."
        )
        print(
            f"Сообщений: {len(all_messages)}"
        )
        print(
            f"Файл: {Config.FULL_HISTORY_ANALYSIS}"
        )

with client:
    client.loop.run_until_complete(main())
import asyncio
from telethon import events

import re

from telegram import Update

from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

from mybot.storage.workspace import (
    create_workspace
)

from mybot.ai.client import generate_answers
from config import Config
from mybot.ai.context_builder import build_ai_request
from mybot.telegram.dialogs import choose_dialog

from mybot.telegram.exporter import (
    append_message_data,
    append_messages_data,
    fetch_messages_after,
    message_to_data
)

from mybot.storage.deletions import (
    check_recent_deletions,
    mark_messages_deleted
)

from mybot.storage.history import (
    get_last_saved_message_id,
    load_all_messages,
    load_last_messages,
    print_messages
)

from incremental_episode import (
    initialize_episode_tracker,
    process_live_episode_message
)

from incremental_memory import (
    update_incremental_memory
)

from memory_manager import load_agent_memory
from mybot.telegram.client import client


RECENT_MESSAGES_LIMIT = 15

def display_message(message):
    text = message.get("text")

    if text:
        content = text
    else:
        content = (
            f'[{message.get("type", "unknown")}]'
        )

    print(
        f'{message.get("sender")}: '
        f'{content}'
    )


async def main():
    current_mood = None
    generation_lock = asyncio.Lock()
    message_processing_lock = (
        asyncio.Lock()
    )

    print(
        "\nВыбери диалог для "
        "real-time режима:\n"
    )

    selected_dialog = await choose_dialog(
        client
    )

    dialog_name = (
        selected_dialog.name
        or f"Dialog {selected_dialog.id}"
    )

    me = await client.get_me()
    workspace = create_workspace(
        account_id=me.id,
        dialog_id=selected_dialog.id,
        dialog_name=dialog_name
    )

    print(
        "\nWorkspace:"
    )

    print(
        workspace.root
    )

    print(
        f"\nВыбран диалог: "
        f"{dialog_name}"
    )

    # ---------------------------------
    # 1. Узнаём последнее сообщение,
    # которое уже сохранено локально
    # ---------------------------------

    last_saved_message_id = (
        get_last_saved_message_id(
            Config.CHAT_HISTORY_JSONL
        )
    )

    print(
        "\nПоследнее сообщение "
        "в локальной истории:"
    )

    print(
        f"ID {last_saved_message_id}"
    )

    # ---------------------------------
    # 2. Догружаем сообщения,
    # которые появились, пока агент
    # был выключен
    # ---------------------------------

    print(
        "\nПроверяю сообщения, "
        "которые появились, "
        "пока агент был выключен..."
    )

    new_messages = (
        await fetch_messages_after(
            client=client,
            dialog=selected_dialog,
            min_message_id=
                last_saved_message_id,
            me_id=me.id
        )
    )

    if new_messages:
        append_messages_data(
            new_messages
        )

        print(
            f"Догружено новых сообщений: "
            f"{len(new_messages)}"
        )

    else:
        print(
            "Новых сообщений нет."
        )

    # ---------------------------------
    # 3. После догрузки читаем
    # актуальные последние сообщения
    # ---------------------------------

    print(
        "\nПроверяю недавние "
        "удалённые сообщения..."
    )

    raw_history = load_all_messages(
        Config.CHAT_HISTORY_JSONL,
        include_deleted=True
    )

    deleted_while_offline = (
        await check_recent_deletions(
            client=client,
            dialog_id=
                selected_dialog.id,
            messages=raw_history
        )
    )

    newly_deleted = (
        mark_messages_deleted(
            deleted_while_offline
        )
    )

    if newly_deleted:
        print(
            f"Найдено удалённых сообщений: "
            f"{len(newly_deleted)}"
        )
    else:
        print(
            "Новых удалённых "
            "сообщений не найдено."
        )

    recent_messages = (
        load_last_messages(
            filename=
                Config.CHAT_HISTORY_JSONL,
            count=RECENT_MESSAGES_LIMIT
        )
    )

    known_message_ids = {
        message.get("message_id")
        for message in recent_messages
        if message.get("message_id")
        is not None
    }

    print(
        "\nТекущий контекст:\n"
    )

    print_messages(
        recent_messages
    )

    memory = load_agent_memory()

    print(
        "\nПроверяю новые эпизоды..."
    )

    episode_tracker, new_episodes = (
        await initialize_episode_tracker()
    )

    if new_episodes:
        print(
            f"Добавлено новых эпизодов: "
            f"{len(new_episodes)}"
        )

        print(
            "Для них созданы только "
            "новые embeddings."
        )

    else:
        print(
            "Новых завершённых "
            "эпизодов нет."
        )


    def apply_mood_to_instruction(
        instruction
    ):
        if not current_mood:
            return instruction

        return f"""
{instruction}

ВРЕМЕННОЕ СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ:

{current_mood}

Это состояние относится только к текущей
сессии и не является постоянным фактом
о пользователе.

Учитывай его при составлении ответа.

ВАЖНО:
- не сообщай состояние напрямую,
  если пользователь сам этого не попросил;
- передавай его через тон, энергию,
  длину, эмоциональность и формулировки;
- сохраняй естественную реакцию
  на сообщения собеседника;
- сохраняй обычный стиль пользователя;
- любовный ответ может оставаться любовным,
  но с указанным эмоциональным оттенком.
""".strip()


    async def generate_bot_answers(
        instruction=None
    ):
        if not instruction:
            instruction = (
                "Ответь естественно, "
                "полностью сохраняя "
                "мой стиль общения."
            )

        instruction = (
            apply_mood_to_instruction(
                instruction
            )
        )

        messages_snapshot = list(
            recent_messages
        )

        async with generation_lock:
            ai_request = (
                await build_ai_request(
                    messages_snapshot,
                    instruction,
                    memory
                )
            )

            with open(
                Config.AI_REQUEST_PREVIEW,
                "w",
                encoding="utf-8"
            ) as file:
                file.write(
                    ai_request
                )

            answers = await generate_answers(
                ai_request
            )

        return answers


    def split_answer_variants(text):
        pattern = (
            r"(?ms)^\s*[123][\.\)]\s*"
            r"(.*?)"
            r"(?=^\s*[123][\.\)]\s*|\Z)"
        )

        variants = [
            item.strip()
            for item in re.findall(
                pattern,
                text
            )
        ]

        if len(variants) == 3:
            return variants

        return [text.strip()]


    def format_messages_for_bot(
        messages
    ):
        lines = []

        for message in messages:
            sender = message.get(
                "sender",
                "?"
            )

            text = message.get(
                "text"
            )

            if not text:
                message_type = (
                    message.get(
                        "type",
                        "unknown"
                    )
                )

                text = (
                    f"[{message_type}]"
                )

            lines.append(
                f"{sender}: {text}"
            )

        return "\n\n".join(
            lines
        )

    def is_owner(update):
        user = update.effective_user
        chat = update.effective_chat

        if not user or not chat:
            return False

        return (
            user.id == me.id
            and chat.id == me.id
            and chat.type == "private"
        )


    async def bot_start(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not is_owner(update):
            return

        await update.message.reply_text(
            "AI Agent запущен.\n\n"
            "/reply — предложить ответ\n"
            "/show — текущий диалог\n"
            "/mood — текущее настроение\n"
            "/mood <текст> — задать настроение\n"
            "/clear_mood — сбросить настроение\n"
            "/update_memory — обновить память\n\n"
            "Также можешь просто написать мне "
            "обычным сообщением, что именно "
            "нужно сказать собеседнику."
        )

    async def send_long_text(
        message,
        text,
        chunk_size=3000
    ):
        for start in range(
            0,
            len(text),
            chunk_size
        ):
            await message.reply_text(
                text[
                    start:
                    start + chunk_size
                ]
            )

    async def bot_show(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not is_owner(update):
            return

        text = format_messages_for_bot(
            recent_messages
        )

        if not text:
            text = "Контекст пуст."

        await send_long_text(
            update.message,
            text
        )


    async def bot_mood(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        nonlocal current_mood

        if not is_owner(update):
            return

        if context.args:
            current_mood = " ".join(
                context.args
            ).strip()

            await update.message.reply_text(
                "Настроение установлено:\n\n"
                f"{current_mood}"
            )

            return

        if current_mood:
            await update.message.reply_text(
                "Текущее настроение:\n\n"
                f"{current_mood}"
            )
        else:
            await update.message.reply_text(
                "Настроение не задано."
            )


    async def bot_clear_mood(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        nonlocal current_mood

        if not is_owner(update):
            return

        current_mood = None

        await update.message.reply_text(
            "Настроение сброшено."
        )


    async def send_generated_answers(
        update,
        instruction=None
    ):
        if not is_owner(update):
            return

        status_message = (
            await update.message.reply_text(
                "Генерирую ответ..."
            )
        )

        try:
            answers = (
                await generate_bot_answers(
                    instruction
                )
            )

        except Exception as error:
            await status_message.edit_text(
                f"Ошибка генерации:\n{error}"
            )

            return

        await status_message.edit_text(
            "3 варианта:"
        )

        variants = split_answer_variants(
            answers
        )

        for variant in variants:
            await send_long_text(
                update.message,
                variant
            )


    async def bot_reply(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        instruction = None

        if context.args:
            instruction = " ".join(
                context.args
            )

        await send_generated_answers(
            update,
            instruction
        )


    async def bot_text_instruction(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not is_owner(update):
            return

        instruction = (
            update.message.text.strip()
        )

        await send_generated_answers(
            update,
            instruction
        )


    async def bot_update_memory(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not is_owner(update):
            return

        message = (
            await update.message.reply_text(
                "Обновляю память..."
            )
        )

        try:
            result = (
                await update_incremental_memory()
            )

        except Exception as error:
            await message.edit_text(
                "Ошибка обновления памяти:\n"
                f"{error}"
            )

            return

        await message.edit_text(
            "Память обновлена.\n\n"
            f"Новых сообщений: "
            f'{result["messages"]}\n'
            f"Новых memories: "
            f'{result["memories"]}'
        )

    if not Config.AGENT_BOT_TOKEN:
        raise ValueError(
            "В .env отсутствует "
            "AGENT_BOT_TOKEN"
        )

    bot_application = (
        Application.builder()
        .token(
            Config.AGENT_BOT_TOKEN
        )
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .build()
    )

    bot_application.add_handler(
        CommandHandler(
            "start",
            bot_start
        )
    )

    bot_application.add_handler(
        CommandHandler(
            "show",
            bot_show
        )
    )

    bot_application.add_handler(
        CommandHandler(
            "reply",
            bot_reply
        )
    )

    bot_application.add_handler(
        CommandHandler(
            "mood",
            bot_mood
        )
    )

    bot_application.add_handler(
        CommandHandler(
            "clear_mood",
            bot_clear_mood
        )
    )

    bot_application.add_handler(
        CommandHandler(
            "update_memory",
            bot_update_memory
        )
    )

    bot_application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            bot_text_instruction
        )
    )

    await bot_application.initialize()
    await bot_application.updater.start_polling()
    await bot_application.start()

    print(
        "\nTelegram-интерфейс "
        "агента запущен."
    )

    async def on_message_deleted(event):
        deleted_ids = set(
            event.deleted_ids
        )

        # MessageDeleted в private chat
        # может не содержать chat_id,
        # поэтому проверяем ID по нашей
        # сохранённой истории.
        local_message_ids = {
            message.get(
                "message_id"
            )
            for message in raw_history
            if message.get(
                "message_id"
            ) is not None
        }

        local_message_ids.update(
            known_message_ids
        )

        relevant_deleted_ids = (
            deleted_ids
            & local_message_ids
        )

        if not relevant_deleted_ids:
            return

        candidate_ids = sorted(
            relevant_deleted_ids
        )

        try:
            telegram_messages = (
                await client.get_messages(
                    selected_dialog.id,
                    ids=candidate_ids
                )
            )

        except Exception as error:
            print(
                "\nНе удалось проверить "
                "удалённые сообщения:"
            )
            print(error)
            return

        confirmed_deleted_ids = {
            message_id
            for message_id, telegram_message
            in zip(
                candidate_ids,
                telegram_messages
            )
            if telegram_message is None
        }

        if not confirmed_deleted_ids:
            return

        newly_deleted = (
            mark_messages_deleted(
                confirmed_deleted_ids
            )
        )

        if not newly_deleted:
            return

        # Убираем сообщение из текущего
        # оперативного контекста.
        recent_messages[:] = [
            message
            for message in recent_messages
            if message.get(
                "message_id"
            ) not in newly_deleted
        ]

        # Убираем его из ещё не завершённого
        # episode tracker.
        episode_tracker.remove_message_ids(
            newly_deleted
        )

        print(
            "\n\n"
            "=========================="
        )

        print(
            "СООБЩЕНИЕ УДАЛЕНО"
        )

        print(
            "=========================="
        )

        for message_id in sorted(
            newly_deleted
        ):
            print(
                f"Message ID: {message_id}"
            )

        print(
            "\nУдалённое сообщение "
            "больше не используется "
            "агентом."
        )


    async def on_new_message(event):
        async with message_processing_lock:
            new_message = (
                await message_to_data(
                    event.message,
                    me.id,
                    dialog_name
                )
            )

            message_id = new_message.get(
                "message_id"
            )

            if (
                message_id
                in known_message_ids
            ):
                return

            if message_id is not None:
                known_message_ids.add(
                    message_id
                )

            # Сразу сохраняем на диск.
            append_message_data(
                new_message
            )

            # Сразу добавляем в текущий
            # контекст независимо от OpenAI.
            recent_messages.append(
                new_message
            )

            if (
                len(recent_messages)
                > RECENT_MESSAGES_LIMIT
            ):
                del recent_messages[
                    :-RECENT_MESSAGES_LIMIT
                ]

            print(
                "\n\n"
                "=========================="
            )

            print(
                "НОВОЕ СООБЩЕНИЕ"
            )

            print(
                "=========================="
            )

            display_message(
                new_message
            )

            # Сначала показываем сообщение
            # пользователю в интерфейс-боте.
            if (
                new_message.get("sender")
                != "Я"
            ):
                try:
                    text = new_message.get(
                        "text"
                    )

                    if not text:
                        message_type = (
                            new_message.get(
                                "type",
                                "unknown"
                            )
                        )

                        text = (
                            f"[{message_type}]"
                        )

                    bot_text = (
                        f"{dialog_name}:\n{text}"
                    )

                    for start in range(
                        0,
                        len(bot_text),
                        3000
                    ):
                        await (
                            bot_application
                            .bot
                            .send_message(
                                chat_id=me.id,
                                text=bot_text[
                                    start:
                                    start + 3000
                                ]
                            )
                        )

                except Exception as error:
                    print(
                        "\nНе удалось отправить "
                        "сообщение в интерфейс "
                        "бота:"
                    )

                    print(error)

            # Обновление episode не должно
            # ломать live-контекст,
            # если embeddings API временно
            # недоступен.
            try:
                new_episode = (
                    await
                    process_live_episode_message(
                        episode_tracker,
                        new_message
                    )
                )

                if new_episode:
                    print(
                        "\nНовый реальный эпизод "
                        "добавлен в память:"
                    )

                    print(
                        f'Episode '
                        f'{new_episode["episode_id"]}'
                    )

            except Exception as error:
                print(
                    "\nОшибка обновления "
                    "episode:"
                )

                print(error)

            print(
                "\nАгент уже видит "
                "и сохранил это сообщение."
            )

    # ---------------------------------
    # 5. Подключаем live listener
    # ---------------------------------

    client.add_event_handler(
        on_new_message,
        events.NewMessage(
            chats=selected_dialog.id
        )
    )

    client.add_event_handler(
        on_message_deleted,
        events.MessageDeleted()
    )

    print(
        "\nREAL-TIME РЕЖИМ ЗАПУЩЕН."
    )

    print(
        "\nТеперь программа постоянно "
        "слушает этот чат."
    )

    print(
        "\nКаждое новое сообщение "
        "автоматически сохраняется "
        "в chat_history.jsonl."
    )

    print(
        "\nЧтобы получить ответ:"
    )

    print(
        "- просто нажми Enter "
        "для обычного ответа"
    )

    print(
        "- или напиши своё указание "
        "и нажми Enter"
    )

    print(
        "- /show — показать "
        "текущий контекст"
    )

    print(
        "- /quit — выйти"
    )

    print(
        "- /mood <описание> — "
        "задать настроение"
    )

    print(
        "- /mood — показать "
        "текущее настроение"
    )

    print(
        "- /clear_mood — "
        "сбросить настроение"
    )

    print(
        "- /update_memory — "
        "обновить долговременную память"
    )

    # ---------------------------------
    # 6. Основной интерфейс агента
    # ---------------------------------

    while True:
        command = await asyncio.to_thread(
            input,
            "\n> "
        )

        command = command.strip()

        if command == "/update_memory":
            print(
                "\nОбновляю долговременную "
                "память..."
            )

            try:
                result = (
                    await update_incremental_memory()
                )

            except Exception as error:
                print(
                    f"\nОшибка обновления памяти: "
                    f"{error}"
                )

                continue

            print(
                f"\nОбработано новых сообщений: "
                f'{result["messages"]}'
            )

            print(
                f"Добавлено новых memories: "
                f'{result["memories"]}'
            )

            continue

        if command == "/mood":
            if current_mood:
                print(
                    "\nТекущее настроение:"
                )
                print(current_mood)
            else:
                print(
                    "\nНастроение не задано."
                )

            continue


        if command.startswith("/mood "):
            current_mood = (
                command[
                    len("/mood "):
                ].strip()
            )

            print(
                "\nНастроение установлено:"
            )
            print(current_mood)

            continue


        if command == "/clear_mood":
            current_mood = None

            print(
                "\nНастроение сброшено."
            )

            continue

        if command == "/quit":
            print(
                "\nReal-time режим "
                "остановлен."
            )

            break

        if command == "/show":
            print(
                "\nТекущий контекст:\n"
            )

            print_messages(
                recent_messages
            )

            continue

        if command:
            user_instruction = command

        else:
            user_instruction = (
                "Ответь естественно, "
                "полностью сохраняя "
                "мой стиль общения."
            )

        if current_mood:
            user_instruction = f"""
{user_instruction}

ВРЕМЕННОЕ СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ:

{current_mood}

Это состояние относится только к текущей
сессии и не является постоянным фактом
о пользователе.

Учитывай его при составлении ответа.

ВАЖНО:
- не сообщай это состояние собеседнику
  напрямую, если пользователь сам
  этого не попросил;
- состояние должно чувствоваться через
  тон, энергию, длину сообщений,
  эмоциональность и формулировки;
- сохраняй естественную реакцию на
  сообщения собеседника;
- не разрушай обычный стиль пользователя;
- если собеседник пишет любовно,
  ответ всё ещё может быть любовным,
  но с указанным эмоциональным оттенком.
""".strip()

        # ВОТ ОТСЮДА УЖЕ НЕ ВНУТРИ if current_mood

        messages_snapshot = list(
            recent_messages
        )

        print(
            "\nПодготавливаю контекст..."
        )

        try:
            ai_request = (
                await build_ai_request(
                    messages_snapshot,
                    user_instruction,
                    memory
                )
            )

            with open(
                Config.AI_REQUEST_PREVIEW,
                "w",
                encoding="utf-8"
            ) as file:
                file.write(
                    ai_request
                )

            print(
                "Отправляю запрос ИИ..."
            )

            answers = (
                await generate_answers(
                    ai_request
                )
            )

        except Exception as error:
            print(
                f"\nОшибка: {error}"
            )

            continue

        print(
            "\nВАРИАНТЫ ОТВЕТА:\n"
        )

        print(
            answers
        )

    print(
        "\nОстанавливаю Telegram-бота..."
    )

    await bot_application.updater.stop()
    await bot_application.stop()
    await bot_application.shutdown()

    print(
        "Telegram-бот остановлен."
    )


with client:
    client.loop.run_until_complete(
        main()
    )
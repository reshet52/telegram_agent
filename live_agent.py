import asyncio
from telethon import events
from mybot.storage.workspace import create_workspace
from mybot.config import Config
from mybot.telegram.dialogs import choose_dialog
from mybot.app.session_state import (
    SessionState
)

from mybot.telegram.bot_interface import (
    BotInterface
)
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
from mybot.episodes.incremental import (
    initialize_episode_tracker,
    process_live_episode_message
)
from mybot.memory.incremental import update_incremental_memory
from mybot.memory.manager import load_agent_memory
from mybot.telegram.client import client
from mybot.services.reply_service import (
    ReplyService
)


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
    state = SessionState()
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

    reply_service = ReplyService(
        recent_messages=recent_messages,
        memory=memory
    )

    bot_interface = BotInterface(
        owner_id=me.id,
        recent_messages=recent_messages,
        reply_service=reply_service,
        state=state
    )

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

    await bot_interface.start()

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
                    await (
                        bot_interface
                        .send_incoming_message(
                            dialog_name,
                            new_message
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
            if state.current_mood:
                print(
                    "\nТекущее настроение:"
                )
                print(state.current_mood)
            else:
                print(
                    "\nНастроение не задано."
                )

            continue


        if command.startswith("/mood "):
            state.current_mood = (
                command[
                    len("/mood "):
                ].strip()
            )

            print(
                "\nНастроение установлено:"
            )
            print(state.current_mood)

            continue


        if command == "/clear_mood":
            state.current_mood = None

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

        print(
            "\nОтправляю запрос ИИ..."
        )

        try:
            answers = (
                await reply_service.generate(
                    instruction=
                        user_instruction,
                    current_mood=
                        state.current_mood
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

    await bot_interface.stop()

with client:
    client.loop.run_until_complete(
        main()
    )
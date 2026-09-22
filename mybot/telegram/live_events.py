import asyncio

from telethon import events

from mybot.episodes.incremental import (
    process_live_episode_message
)
from mybot.storage.deletions import (
    mark_messages_deleted
)
from mybot.telegram.exporter import (
    append_message_data,
    message_to_data
)


class LiveEvents:
    def __init__(
        self,
        client,
        selected_dialog,
        me,
        dialog_name,
        workspace,
        raw_history,
        recent_messages,
        known_message_ids,
        bot_interface,
        episode_tracker,
        recent_messages_limit=15
    ):
        self.client = client
        self.selected_dialog = (
            selected_dialog
        )
        self.me = me
        self.dialog_name = dialog_name
        self.workspace = workspace

        self.raw_history = raw_history
        self.recent_messages = (
            recent_messages
        )
        self.known_message_ids = (
            known_message_ids
        )

        self.bot_interface = (
            bot_interface
        )

        self.episode_tracker = (
            episode_tracker
        )

        self.recent_messages_limit = (
            recent_messages_limit
        )

        self.message_processing_lock = (
            asyncio.Lock()
        )


    def display_message(
        self,
        message
    ):
        text = message.get(
            "text"
        )

        if text:
            content = text

        else:
            content = (
                f'['
                f'{message.get("type", "unknown")}'
                f']'
            )

        print(
            f'{message.get("sender")}: '
            f'{content}'
        )


    async def on_message_deleted(
        self,
        event
    ):
        deleted_ids = set(
            event.deleted_ids
        )

        local_message_ids = {
            message.get(
                "message_id"
            )
            for message
            in self.raw_history
            if message.get(
                "message_id"
            ) is not None
        }

        local_message_ids.update(
            self.known_message_ids
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
                await self.client.get_messages(
                    self.selected_dialog.id,
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
            for (
                message_id,
                telegram_message
            )
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
                confirmed_deleted_ids,
                filename=
                    self.workspace
                    .deleted_message_ids
            )
        )

        if not newly_deleted:
            return

        self.recent_messages[:] = [
            message
            for message
            in self.recent_messages
            if message.get(
                "message_id"
            ) not in newly_deleted
        ]

        self.episode_tracker.remove_message_ids(
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
                f"Message ID: "
                f"{message_id}"
            )

        print(
            "\nУдалённое сообщение "
            "больше не используется "
            "агентом."
        )


    async def on_new_message(
        self,
        event
    ):
        async with (
            self.message_processing_lock
        ):
            new_message = (
                await message_to_data(
                    event.message,
                    self.me.id,
                    self.dialog_name
                )
            )

            message_id = (
                new_message.get(
                    "message_id"
                )
            )

            if (
                message_id
                in self.known_message_ids
            ):
                return

            if message_id is not None:
                self.known_message_ids.add(
                    message_id
                )

            append_message_data(
                new_message,
                filename=
                    self.workspace
                    .chat_history
            )

            self.recent_messages.append(
                new_message
            )

            if (
                len(self.recent_messages)
                > self.recent_messages_limit
            ):
                del self.recent_messages[
                    :-self.recent_messages_limit
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

            self.display_message(
                new_message
            )

            if (
                new_message.get("sender")
                != "Я"
            ):
                try:
                    await (
                        self.bot_interface
                        .send_incoming_message(
                            self.dialog_name,
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

            try:
                new_episode = (
                    await
                    process_live_episode_message(
                        self.episode_tracker,
                        new_message,
                        episodes_filename=
                            self.workspace
                            .episodes,
                        live_embeddings_filename=
                            self.workspace
                            .episode_embeddings_live
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


    def register(
        self
    ):
        self.client.add_event_handler(
            self.on_new_message,
            events.NewMessage(
                chats=
                    self.selected_dialog.id
            )
        )

        self.client.add_event_handler(
            self.on_message_deleted,
            events.MessageDeleted()
        )
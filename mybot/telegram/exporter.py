import json

from mybot.config import Config


def get_file_data(message):
    if not message.file:
        return None

    return {
        "name": getattr(
            message.file,
            "name",
            None
        ),
        "mime_type": getattr(
            message.file,
            "mime_type",
            None
        ),
        "size": getattr(
            message.file,
            "size",
            None
        ),
        "duration": getattr(
            message.file,
            "duration",
            None
        )
    }


def get_message_type(message):
    if message.voice:
        return "voice"

    if message.gif:
        return "gif"

    if message.photo:
        return "photo"

    if message.video:
        return "video"

    if message.sticker:
        return "sticker"

    if message.document:
        return "file"

    return "text"


async def message_to_data(
    message,
    me_id,
    dialog_name
):
    if message.sender_id == me_id:
        sender = "Я"
    else:
        sender = dialog_name

    reply_to_text = None
    reply_to_sender = None

    if message.reply_to_msg_id:
        try:
            replied_message = (
                await message.get_reply_message()
            )

            if replied_message:
                reply_to_text = (
                    replied_message.text
                    or None
                )

                if (
                    replied_message.sender_id
                    == me_id
                ):
                    reply_to_sender = "Я"
                else:
                    reply_to_sender = (
                        dialog_name
                    )

        except Exception:
            pass

    return {
        "message_id": message.id,
        "date": (
            message.date.isoformat()
            if message.date
            else None
        ),
        "sender_id": message.sender_id,
        "sender": sender,
        "type": get_message_type(
            message
        ),
        "text": (
            message.text
            or None
        ),
        "file": get_file_data(
            message
        ),
        "reply_to_message_id":
            message.reply_to_msg_id,
        "reply_to_sender":
            reply_to_sender,
        "reply_to_text":
            reply_to_text
    }


def write_message_data(
    file,
    message_data
):
    json_line = json.dumps(
        message_data,
        ensure_ascii=False,
        indent=4
    )

    file.write(json_line)
    file.write("\n\n")


def append_message_data(
    message_data,
    filename=Config.CHAT_HISTORY_JSONL
):
    with open(
        filename,
        "a",
        encoding="utf-8"
    ) as file:
        write_message_data(
            file,
            message_data
        )


def append_messages_data(
    messages,
    filename=Config.CHAT_HISTORY_JSONL
):
    if not messages:
        return

    with open(
        filename,
        "a",
        encoding="utf-8"
    ) as file:
        for message_data in messages:
            write_message_data(
                file,
                message_data
            )


async def fetch_messages_after(
    client,
    dialog,
    min_message_id,
    me_id
):
    new_messages = []

    async for message in client.iter_messages(
        dialog.id,
        limit=None,
        min_id=min_message_id,
        reverse=True
    ):
        message_data = (
            await message_to_data(
                message,
                me_id,
                dialog.name
            )
        )

        new_messages.append(
            message_data
        )

    return new_messages


async def export_dialog(
    client,
    dialog,
    filename=Config.CHAT_HISTORY_JSONL
):
    me = await client.get_me()

    message_count = 0

    print(
        f"\nВыбран диалог: "
        f"{dialog.name}"
    )

    print(
        f"ID диалога: "
        f"{dialog.id}"
    )

    print(
        "Начинаю загрузку и "
        "сохранение истории..."
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        async for message in (
            client.iter_messages(
                dialog.id,
                limit=None,
                reverse=True
            )
        ):
            message_data = (
                await message_to_data(
                    message,
                    me.id,
                    dialog.name
                )
            )

            write_message_data(
                file,
                message_data
            )

            message_count += 1

            if message_count % 1000 == 0:
                print(
                    f"Сохранено сообщений: "
                    f"{message_count}"
                )

    print(
        f"\nГотово. Всего сохранено "
        f"сообщений: {message_count}"
    )

    print(
        f"История записана "
        f"в файл {filename}"
    )
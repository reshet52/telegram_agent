import json
from collections import deque

from mybot.config import Config
from mybot.storage.deletions import (
    load_deleted_message_ids
)


def load_last_messages(
    filename=Config.CHAT_HISTORY_JSONL,
    count=10,
    include_deleted=False
):
    last_messages = deque(
        maxlen=count
    )

    message_lines = []

    if include_deleted:
        deleted_ids = set()
    else:
        deleted_ids = (
            load_deleted_message_ids()
        )

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                message_lines.append(
                    line
                )

                continue

            if message_lines:
                message_text = "".join(
                    message_lines
                )

                message_data = json.loads(
                    message_text
                )

                message_id = (
                    message_data.get(
                        "message_id"
                    )
                )

                if (
                    include_deleted
                    or message_id
                    not in deleted_ids
                ):
                    last_messages.append(
                        message_data
                    )

                message_lines = []

        if message_lines:
            message_text = "".join(
                message_lines
            )

            message_data = json.loads(
                message_text
            )

            message_id = (
                message_data.get(
                    "message_id"
                )
            )

            if (
                include_deleted
                or message_id
                not in deleted_ids
            ):
                last_messages.append(
                    message_data
                )

    return list(
        last_messages
    )


def get_last_saved_message_id(
    filename=Config.CHAT_HISTORY_JSONL
):
    # Здесь специально читаем даже
    # удалённые записи.
    # Нам нужен настоящий последний ID
    # физически сохранённого сообщения,
    # чтобы не скачать его повторно.
    messages = load_last_messages(
        filename=filename,
        count=1,
        include_deleted=True
    )

    if not messages:
        return 0

    return (
        messages[-1].get(
            "message_id"
        )
        or 0
    )


def print_messages(messages):
    print(
        "\nПоследние сообщения:\n"
    )

    for message in messages:
        sender = message["sender"]
        text = message["text"]
        message_type = (
            message["type"]
        )

        if message_type == "text":
            content = (
                text
                or "[ПУСТОЕ СООБЩЕНИЕ]"
            )

        elif text:
            content = (
                f"[{message_type.upper()}] "
                f"{text}"
            )

        else:
            content = (
                f"[{message_type.upper()}]"
            )

        print(
            f"{sender}: {content}"
        )


def load_all_messages(
    filename=Config.CHAT_HISTORY_JSONL,
    include_deleted=False
):
    messages = []
    message_lines = []

    if include_deleted:
        deleted_ids = set()
    else:
        deleted_ids = (
            load_deleted_message_ids()
        )

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            if line.strip():
                message_lines.append(
                    line
                )

                continue

            if message_lines:
                message_text = "".join(
                    message_lines
                )

                message_data = json.loads(
                    message_text
                )

                message_id = (
                    message_data.get(
                        "message_id"
                    )
                )

                if (
                    include_deleted
                    or message_id
                    not in deleted_ids
                ):
                    messages.append(
                        message_data
                    )

                message_lines = []

        if message_lines:
            message_text = "".join(
                message_lines
            )

            message_data = json.loads(
                message_text
            )

            message_id = (
                message_data.get(
                    "message_id"
                )
            )

            if (
                include_deleted
                or message_id
                not in deleted_ids
            ):
                messages.append(
                    message_data
                )

    return messages


def split_messages_into_chunks(
    messages,
    chunk_size=500
):
    chunks = []

    for start in range(
        0,
        len(messages),
        chunk_size
    ):
        chunk = messages[
            start:
            start + chunk_size
        ]

        chunks.append(
            chunk
        )

    return chunks
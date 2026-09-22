import json
from pathlib import Path

from telethon.errors import MessageIdInvalidError


DEFAULT_DELETED_MESSAGES_FILE = Path(
    "exports/deleted_message_ids.json"
)

RECONCILE_MESSAGES_LIMIT = 200
CHECK_BATCH_SIZE = 100


def resolve_deleted_file(
    filename=None
):
    if filename is None:
        return DEFAULT_DELETED_MESSAGES_FILE

    return Path(filename)


def load_deleted_message_ids(
    filename=None
):
    deleted_file = (
        resolve_deleted_file(
            filename
        )
    )

    if not deleted_file.exists():
        return set()

    with open(
        deleted_file,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        return set()

    return {
        int(message_id)
        for message_id in data
    }


def save_deleted_message_ids(
    message_ids,
    filename=None
):
    deleted_file = (
        resolve_deleted_file(
            filename
        )
    )

    deleted_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        deleted_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            sorted(message_ids),
            file,
            ensure_ascii=False,
            indent=2
        )


def mark_messages_deleted(
    message_ids,
    filename=None
):
    deleted_ids = (
        load_deleted_message_ids(
            filename
        )
    )

    new_deleted_ids = set()

    for message_id in message_ids:
        if message_id is None:
            continue

        message_id = int(
            message_id
        )

        if message_id in deleted_ids:
            continue

        deleted_ids.add(
            message_id
        )

        new_deleted_ids.add(
            message_id
        )

    if new_deleted_ids:
        save_deleted_message_ids(
            deleted_ids,
            filename
        )

    return new_deleted_ids


async def check_recent_deletions(
    client,
    dialog_id,
    messages,
    limit=RECONCILE_MESSAGES_LIMIT,
    deleted_filename=None
):
    recent_messages = messages[
        -limit:
    ]

    message_ids = [
        message.get("message_id")
        for message in recent_messages
        if message.get(
            "message_id"
        ) is not None
    ]

    already_deleted = (
        load_deleted_message_ids(
            deleted_filename
        )
    )

    message_ids = [
        message_id
        for message_id in message_ids
        if message_id not in already_deleted
    ]

    deleted_ids = set()

    for start in range(
        0,
        len(message_ids),
        CHECK_BATCH_SIZE
    ):
        batch_ids = message_ids[
            start:
            start + CHECK_BATCH_SIZE
        ]

        try:
            telegram_messages = (
                await client.get_messages(
                    dialog_id,
                    ids=batch_ids
                )
            )

        except MessageIdInvalidError:
            # На всякий случай проверяем
            # сообщения по одному.
            telegram_messages = []

            for message_id in batch_ids:
                try:
                    telegram_message = (
                        await client.get_messages(
                            dialog_id,
                            ids=message_id
                        )
                    )

                except MessageIdInvalidError:
                    telegram_message = None

                telegram_messages.append(
                    telegram_message
                )

        # При ids=[...] Telethon сохраняет
        # соответствие позиции ID и результата.
        for (
            message_id,
            telegram_message
        ) in zip(
            batch_ids,
            telegram_messages
        ):
            if telegram_message is None:
                deleted_ids.add(
                    message_id
                )

    return deleted_ids
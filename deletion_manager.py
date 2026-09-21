import json
from pathlib import Path

from telethon.errors import MessageIdInvalidError


DELETED_MESSAGES_FILE = Path(
    "exports/deleted_message_ids.json"
)

RECONCILE_MESSAGES_LIMIT = 200
CHECK_BATCH_SIZE = 100


def load_deleted_message_ids():
    if not DELETED_MESSAGES_FILE.exists():
        return set()

    with open(
        DELETED_MESSAGES_FILE,
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
    message_ids
):
    with open(
        DELETED_MESSAGES_FILE,
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
    message_ids
):
    deleted_ids = (
        load_deleted_message_ids()
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
            deleted_ids
        )

    return new_deleted_ids


async def check_recent_deletions(
    client,
    dialog_id,
    messages,
    limit=RECONCILE_MESSAGES_LIMIT
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
        load_deleted_message_ids()
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
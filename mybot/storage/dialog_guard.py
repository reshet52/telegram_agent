from pathlib import Path

from mybot.config import Config
from mybot.storage.history import (
    load_all_messages
)


def validate_legacy_dialog(
    me_id,
    selected_dialog_id,
    history_filename=Config.CHAT_HISTORY_JSONL
):
    history_path = Path(
        history_filename
    )

    if not history_path.exists():
        return True, set()

    messages = load_all_messages(
        filename=history_path,
        include_deleted=True
    )

    other_sender_ids = {
        message.get("sender_id")
        for message in messages
        if (
            message.get("sender_id")
            is not None
            and message.get("sender_id")
            != me_id
        )
    }

    # Пустая история ничего
    # не ограничивает.
    if not other_sender_ids:
        return True, set()

    is_valid = (
        other_sender_ids
        == {selected_dialog_id}
    )

    return (
        is_valid,
        other_sender_ids
    )
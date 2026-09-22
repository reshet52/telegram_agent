import argparse
import shutil
from pathlib import Path

from mybot.config import Config
from mybot.storage.dialog_guard import (
    validate_legacy_dialog
)
from mybot.storage.workspace import (
    create_workspace
)


LEGACY_FILES = [
    (
        Path(Config.CHAT_HISTORY_JSONL),
        "chat_history"
    ),
    (
        Path(
            "exports/deleted_message_ids.json"
        ),
        "deleted_message_ids"
    ),
    (
        Path("memory/episodes.jsonl"),
        "episodes"
    ),
    (
        Path(
            "memory/episode_embeddings.json"
        ),
        "episode_embeddings"
    ),
    (
        Path(
            "memory/"
            "episode_embeddings_live.jsonl"
        ),
        "episode_embeddings_live"
    ),
    (
        Path("memory/agent_memory.json"),
        "agent_memory"
    ),
    (
        Path(
            "memory/agent_memory_live.jsonl"
        ),
        "agent_memory_live"
    ),
    (
        Path(
            "memory/memory_embeddings.json"
        ),
        "memory_embeddings"
    ),
    (
        Path(
            "memory/"
            "memory_embeddings_live.jsonl"
        ),
        "memory_embeddings_live"
    ),
    (
        Path(
            "memory/memory_update_state.json"
        ),
        "memory_update_state"
    ),
    (
        Path(Config.USER_PROFILE),
        "user_profile"
    ),
    (
        Path(Config.PERSON_PROFILE),
        "person_profile"
    )
]


def copy_file(
    source,
    destination
):
    if not source.exists():
        print(
            f"Нет исходного файла: "
            f"{source}"
        )
        return

    if destination.exists():
        print(
            f"Уже существует, пропускаю: "
            f"{destination}"
        )
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    print(
        f"Скопировано: "
        f"{source}"
        f" -> "
        f"{destination}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--account-id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--dialog-id",
        type=int,
        required=True
    )

    parser.add_argument(
        "--dialog-name",
        required=True
    )

    args = parser.parse_args()

    is_valid, legacy_dialog_ids = (
        validate_legacy_dialog(
            me_id=args.account_id,
            selected_dialog_id=
                args.dialog_id,
            history_filename=
                Config.CHAT_HISTORY_JSONL
        )
    )

    if not is_valid:
        raise RuntimeError(
            "Legacy history принадлежит "
            "другому диалогу. "
            "Миграция остановлена. "
            f"Найденные dialog ID: "
            f"{sorted(legacy_dialog_ids)}"
        )

    workspace = create_workspace(
        account_id=args.account_id,
        dialog_id=args.dialog_id,
        dialog_name=args.dialog_name
    )

    print(
        "\nWorkspace:"
    )

    print(
        workspace.root
    )

    print(
        "\nНачинаю копирование...\n"
    )

    for source, property_name in (
        LEGACY_FILES
    ):
        destination = getattr(
            workspace,
            property_name
        )

        copy_file(
            source,
            destination
        )

    print(
        "\nМиграция завершена."
    )

    print(
        "Исходные legacy-файлы "
        "не изменялись."
    )


if __name__ == "__main__":
    main()
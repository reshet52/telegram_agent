import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path("data")


@dataclass
class Workspace:
    account_id: int
    dialog_id: int
    dialog_name: str
    root: Path

    @property
    def manifest(self):
        return (
            self.root
            / "workspace.json"
        )

    @property
    def chat_history(self):
        return (
            self.root
            / "chat_history.jsonl"
        )

    @property
    def deleted_message_ids(self):
        return (
            self.root
            / "deleted_message_ids.json"
        )

    @property
    def ai_request_preview(self):
        return (
            self.root
            / "ai_request_preview.txt"
        )

    @property
    def reply_journal(self):
        return self.root / "reply_journal.sqlite3"

    @property
    def full_history_analysis(self):
        return (
            self.root
            / "full_history_analysis.txt"
        )

    @property
    def episodes(self):
        return (
            self.root
            / "episodes.jsonl"
        )

    @property
    def episode_embeddings(self):
        return (
            self.root
            / "episode_embeddings.json"
        )

    @property
    def episode_embeddings_live(self):
        return (
            self.root
            / "episode_embeddings_live.jsonl"
        )

    @property
    def agent_memory(self):
        return (
            self.root
            / "agent_memory.json"
        )

    @property
    def agent_memory_live(self):
        return (
            self.root
            / "agent_memory_live.jsonl"
        )

    @property
    def memory_embeddings(self):
        return (
            self.root
            / "memory_embeddings.json"
        )

    @property
    def memory_embeddings_live(self):
        return (
            self.root
            / "memory_embeddings_live.jsonl"
        )

    @property
    def memory_update_state(self):
        return (
            self.root
            / "memory_update_state.json"
        )

    @property
    def user_profile(self):
        return (
            self.root
            / "user_profile.json"
        )

    @property
    def person_profile(self):
        return (
            self.root
            / "person_profile.json"
        )

    @property
    def chunk_analysis(self):
        return (
            self.root
            / "chunk_analysis"
        )


def get_account_root(
    account_id
):
    return (
        DATA_ROOT
        / f"account_{account_id}"
    )


def get_global_root(
    account_id
):
    return (
        get_account_root(
            account_id
        )
        / "global"
    )


def get_dialogs_root(
    account_id
):
    return (
        get_account_root(
            account_id
        )
        / "dialogs"
    )


def get_workspace_root(
    account_id,
    dialog_id
):
    return (
        get_dialogs_root(
            account_id
        )
        / f"dialog_{dialog_id}"
    )


def create_workspace(
    account_id,
    dialog_id,
    dialog_name
):
    root = get_workspace_root(
        account_id,
        dialog_id
    )

    root.mkdir(
        parents=True,
        exist_ok=True
    )

    chunk_analysis = (
        root
        / "chunk_analysis"
    )

    chunk_analysis.mkdir(
        parents=True,
        exist_ok=True
    )

    global_root = get_global_root(
        account_id
    )

    global_root.mkdir(
        parents=True,
        exist_ok=True
    )

    workspace = Workspace(
        account_id=account_id,
        dialog_id=dialog_id,
        dialog_name=dialog_name,
        root=root
    )

    save_workspace_manifest(
        workspace
    )

    return workspace


def save_workspace_manifest(
    workspace
):
    existing_data = {}

    if workspace.manifest.exists():
        try:
            with open(
                workspace.manifest,
                "r",
                encoding="utf-8"
            ) as file:
                existing_data = (
                    json.load(file)
                )

        except (
            json.JSONDecodeError,
            OSError
        ):
            existing_data = {}

    created_at = (
        existing_data.get(
            "created_at"
        )
    )

    if not created_at:
        created_at = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    data = {
        "account_id":
            workspace.account_id,

        "dialog_id":
            workspace.dialog_id,

        "dialog_name":
            workspace.dialog_name,

        "created_at":
            created_at,

        "updated_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    with open(
        workspace.manifest,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


def load_workspace(
    account_id,
    dialog_id
):
    root = get_workspace_root(
        account_id,
        dialog_id
    )

    manifest = (
        root
        / "workspace.json"
    )

    if not manifest.exists():
        return None

    with open(
        manifest,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return Workspace(
        account_id=data[
            "account_id"
        ],
        dialog_id=data[
            "dialog_id"
        ],
        dialog_name=data.get(
            "dialog_name",
            f"Dialog {dialog_id}"
        ),
        root=root
    )


def list_workspaces(
    account_id
):
    dialogs_root = (
        get_dialogs_root(
            account_id
        )
    )

    if not dialogs_root.exists():
        return []

    workspaces = []

    for folder in (
        dialogs_root.iterdir()
    ):
        if not folder.is_dir():
            continue

        manifest = (
            folder
            / "workspace.json"
        )

        if not manifest.exists():
            continue

        try:
            with open(
                manifest,
                "r",
                encoding="utf-8"
            ) as file:
                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):
            continue

        workspace = Workspace(
            account_id=data[
                "account_id"
            ],
            dialog_id=data[
                "dialog_id"
            ],
            dialog_name=data.get(
                "dialog_name",
                "Unknown"
            ),
            root=folder
        )

        workspaces.append(
            workspace
        )

    return workspaces

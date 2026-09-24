"""Local allowlist and distinct Telegram API credentials for each owner."""

import json
import os
from dataclasses import dataclass

from mybot.config import Config
from mybot.storage.workspace import get_global_root


@dataclass(frozen=True)
class AccountCredentials:
    owner_id: int
    api_id: int
    api_hash: str
    session: str


def load_accounts():
    raw = os.getenv("ALLOWED_TELEGRAM_IDS", "").strip()
    if not raw:
        raise ValueError("Укажите ALLOWED_TELEGRAM_IDS в .env (ID через запятую).")
    ids = [int(part.strip()) for part in raw.split(",")]
    if len(ids) != len(set(ids)) or any(owner_id <= 0 for owner_id in ids):
        raise ValueError("ALLOWED_TELEGRAM_IDS содержит дубликаты или неверный ID.")
    primary_id = int(os.getenv("PRIMARY_TELEGRAM_ID", ids[0]))
    if primary_id not in ids:
        raise ValueError("PRIMARY_TELEGRAM_ID должен входить в ALLOWED_TELEGRAM_IDS.")
    accounts = {}
    for owner_id in ids:
        path = get_global_root(owner_id) / "telegram_credentials.json"
        if owner_id == primary_id:
            # Always retain the pre-existing primary .session path.
            api_id, api_hash, session = Config.API_ID, Config.API_HASH, Config.SESSION_NAME
        elif path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            api_id = int(data["api_id"])
            api_hash = data["api_hash"]
            session = str(get_global_root(owner_id) / "telegram")
        else:
            raise ValueError(f"Нет учётных данных для аккаунта {owner_id}: {path}")
        if not api_id or not api_hash:
            raise ValueError(f"Не заданы API_ID/API_HASH для {owner_id}")
        accounts[owner_id] = AccountCredentials(owner_id, api_id, api_hash, session)
    sessions = [item.session for item in accounts.values()]
    pairs = [(item.api_id, item.api_hash) for item in accounts.values()]
    if len(sessions) != len(set(sessions)) or len(pairs) != len(set(pairs)):
        raise ValueError("У аккаунтов должны быть разные API-пары и файлы сессий.")
    return accounts

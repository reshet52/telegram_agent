"""Small, account-local state of the private control interface."""

from mybot.storage.atomic import read_json, write_json
from mybot.storage.workspace import get_global_root


class ControlState:
    def __init__(self, account_id):
        self.path = get_global_root(account_id) / "control_state.json"
        self.value = read_json(self.path, {})
        if not isinstance(self.value, dict):
            raise ValueError(f"Invalid control state: {self.path}")

    def set(self, **changes):
        updated = {**self.value, **changes}
        write_json(self.path, updated)
        self.value = updated

    def get(self, key, default=None):
        return self.value.get(key, default)

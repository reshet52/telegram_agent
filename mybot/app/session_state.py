from dataclasses import dataclass


@dataclass
class SessionState:
    current_mood: str | None = None
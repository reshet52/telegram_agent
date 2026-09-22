import json
from pathlib import Path

from mybot.config import Config


def load_json_file(filename):
    file_path = Path(filename)

    if not file_path.exists():
        return {}

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def load_agent_memory(
    user_profile_filename=
        Config.USER_PROFILE,
    person_profile_filename=
        Config.PERSON_PROFILE
):
    user_profile = load_json_file(
        user_profile_filename
    )

    person_profile = load_json_file(
        person_profile_filename
    )

    return {
        "user_profile":
            user_profile,
        "person_profile":
            person_profile
    }
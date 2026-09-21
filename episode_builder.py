import json
from datetime import datetime
from pathlib import Path

from config import Config
from mybot.storage.history import load_all_messages


OUTPUT_FILE = Path("memory/episodes.jsonl")
PREVIOUS_CONTEXT_LIMIT = 20


def group_messages_by_sender(messages):
    groups = []

    for message in messages:
        sender = message.get(
            "sender",
            "Неизвестно"
        )

        if (
            groups
            and groups[-1]["sender"] == sender
        ):
            groups[-1]["messages"].append(
                message
            )
        else:
            groups.append({
                "sender": sender,
                "messages": [message]
            })

    return groups


def get_previous_context(
    groups,
    current_index,
    limit=PREVIOUS_CONTEXT_LIMIT
):
    context = []

    # Идём назад от incoming-блока
    group_index = current_index - 2

    while (
        group_index >= 0
        and len(context) < limit
    ):
        group_messages = groups[
            group_index
        ]["messages"]

        # Добавляем сообщения с конца,
        # потому что идём назад по истории
        for message in reversed(
            group_messages
        ):
            context.append(message)

            if len(context) >= limit:
                break

        group_index -= 1

    # Возвращаем нормальный
    # хронологический порядок
    context.reverse()

    return context


def parse_date(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (ValueError, TypeError):
        return None


def calculate_response_delay(
    incoming,
    response
):
    if not incoming or not response:
        return None

    incoming_date = parse_date(
        incoming[-1].get("date")
    )

    response_date = parse_date(
        response[0].get("date")
    )

    if (
        incoming_date is None
        or response_date is None
    ):
        return None

    delay = (
        response_date - incoming_date
    ).total_seconds()

    if delay < 0:
        return None

    return int(delay)


def build_episodes(groups):
    episodes = []

    for index, group in enumerate(groups):
        # Нас интересуют блоки,
        # написанные пользователем
        if group["sender"] != "Я":
            continue

        if index == 0:
            continue

        incoming_group = groups[index - 1]

        # Перед ответом должен находиться
        # блок другого собеседника
        if incoming_group["sender"] == "Я":
            continue

        previous_context = (
            get_previous_context(
                groups,
                index
            )
        )

        incoming = (
            incoming_group["messages"]
        )

        response = group["messages"]

        episode = {
            "episode_id": len(episodes) + 1,

            "previous_context":
                previous_context,

            "incoming":
                incoming,

            "response":
                response,

            "metadata": {
                "incoming_message_count":
                    len(incoming),

                "response_message_count":
                    len(response),

                "incoming_started_at":
                    incoming[0].get("date")
                    if incoming
                    else None,

                "incoming_ended_at":
                    incoming[-1].get("date")
                    if incoming
                    else None,

                "response_started_at":
                    response[0].get("date")
                    if response
                    else None,

                "response_ended_at":
                    response[-1].get("date")
                    if response
                    else None,

                "response_delay_seconds":
                    calculate_response_delay(
                        incoming,
                        response
                    )
            }
        }

        episodes.append(episode)

    return episodes


def save_episodes(
    episodes,
    filename=OUTPUT_FILE
):
    filename.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:
        for episode in episodes:
            json.dump(
                episode,
                file,
                ensure_ascii=False
            )

            file.write("\n")


def print_episode_summary(episode):
    print(
        f'Episode {episode["episode_id"]}: '
        f'{episode["metadata"]["incoming_message_count"]} '
        f'→ '
        f'{episode["metadata"]["response_message_count"]} '
        f'сообщений'
    )

    delay = episode[
        "metadata"
    ]["response_delay_seconds"]

    if delay is not None:
        print(
            f"  Задержка ответа: "
            f"{delay} сек."
        )


def main():
    messages = load_all_messages(
        Config.CHAT_HISTORY_JSONL
    )

    print(
        f"Всего сообщений: "
        f"{len(messages)}"
    )

    groups = group_messages_by_sender(
        messages
    )

    episodes = build_episodes(
        groups
    )

    print(
        f"Получилось эпизодов: "
        f"{len(episodes)}"
    )

    save_episodes(
        episodes
    )

    print(
        f"\nБаза сохранена:"
    )
    print(OUTPUT_FILE)

    print(
        "\nПоследние 10 эпизодов:"
    )

    for episode in episodes[-10:]:
        print_episode_summary(
            episode
        )


if __name__ == "__main__":
    main()
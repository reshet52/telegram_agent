import json

from episode_search import find_similar_episodes
from memory_search import find_relevant_memories


def build_dialog_context(messages):
    context_lines = []

    for message in messages:
        sender = message.get(
            "sender",
            "Неизвестно"
        )

        text = message.get("text")
        message_type = message.get(
            "type",
            "text"
        )

        if message_type == "text":
            content = (
                text
                or "[ПУСТОЕ СООБЩЕНИЕ]"
            )

        elif text:
            content = (
                f"[{message_type.upper()}] "
                f"{text}"
            )

        else:
            content = (
                f"[{message_type.upper()}]"
            )

        reply_to_text = message.get(
            "reply_to_text"
        )

        reply_to_sender = message.get(
            "reply_to_sender"
        )

        if reply_to_text:
            context_lines.append(
                f'{sender} отвечает на '
                f'сообщение от '
                f'{reply_to_sender}: '
                f'"{reply_to_text}"'
            )

        context_lines.append(
            f"{sender}: {content}"
        )

    return "\n".join(
        context_lines
    )


def format_memory(memory):
    return json.dumps(
        memory,
        ensure_ascii=False,
        indent=2
    )


def format_relevant_memories(results):
    if not results:
        return (
            "Подходящей долговременной "
            "памяти не найдено."
        )

    sections = []

    for number, result in enumerate(
        results,
        start=1
    ):
        memory = result["memory"]
        score = result["score"]

        sections.append(
            "\n".join([
                f"ПАМЯТЬ {number}",
                (
                    f'Категория: '
                    f'{memory["category"]}'
                ),
                (
                    f'Similarity: '
                    f'{score:.4f}'
                ),
                (
                    f'Time context: '
                    f'{memory["time_context"]}'
                ),
                (
                    f'Confidence: '
                    f'{memory["confidence"]}'
                ),
                "",
                memory["memory"]
            ])
        )

    return "\n\n".join(
        sections
    )


def get_current_incoming(messages):
    incoming = []

    for message in reversed(messages):
        sender = message.get("sender")

        if sender == "Я":
            break

        incoming.append(message)

    incoming.reverse()

    return incoming


def format_episode_message(message):
    sender = message.get(
        "sender",
        "Неизвестно"
    )

    text = message.get("text")

    if text:
        content = text
    else:
        message_type = message.get(
            "type",
            "unknown"
        )

        content = (
            f"[{message_type.upper()}]"
        )

    return f"{sender}: {content}"


def format_similar_episodes(results):
    if not results:
        return (
            "Подходящих исторических "
            "эпизодов не найдено."
        )

    sections = []

    for number, result in enumerate(
        results,
        start=1
    ):
        episode = result["episode"]
        score = result["score"]

        lines = [
            (
                f"ИСТОРИЧЕСКИЙ ПРИМЕР "
                f"{number}"
            ),
            (
                f"Similarity: "
                f"{score:.4f}"
            ),
            "",
            "Контекст перед ситуацией:"
        ]

        previous_context = episode.get(
            "previous_context",
            []
        )[-8:]

        for message in previous_context:
            lines.append(
                format_episode_message(
                    message
                )
            )

        lines.append("")
        lines.append(
            "Сообщения собеседника:"
        )

        for message in episode.get(
            "incoming",
            []
        ):
            lines.append(
                format_episode_message(
                    message
                )
            )

        lines.append("")
        lines.append(
            "Реальный ответ пользователя:"
        )

        for message in episode.get(
            "response",
            []
        ):
            lines.append(
                format_episode_message(
                    message
                )
            )

        sections.append(
            "\n".join(lines)
        )

    return "\n\n".join(
        sections
    )


async def build_ai_request(
    messages,
    user_instruction,
    memory
):
    dialog_context = (
        build_dialog_context(
            messages
        )
    )

    user_profile = format_memory(
        memory.get(
            "user_profile",
            {}
        )
    )

    person_profile = format_memory(
        memory.get(
            "person_profile",
            {}
        )
    )

    current_incoming = (
        get_current_incoming(
            messages
        )
    )

    relevant_memories = (
        await find_relevant_memories(
            messages
        )
    )

    relevant_memory_text = (
        format_relevant_memories(
            relevant_memories
        )
    )

    if current_incoming:
        similar_episodes = (
            await find_similar_episodes(
                messages
            )
        )
    else:
        similar_episodes = []

    historical_examples = (
        format_similar_episodes(
            similar_episodes
        )
    )

    return f"""
ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:

{user_profile}


ПРОФИЛЬ СОБЕСЕДНИКА:

{person_profile}


РЕЛЕВАНТНАЯ ДОЛГОВРЕМЕННАЯ ПАМЯТЬ:

{relevant_memory_text}


ПОХОЖИЕ ИСТОРИЧЕСКИЕ СИТУАЦИИ:

{historical_examples}


ПОСЛЕДНИЕ СООБЩЕНИЯ ТЕКУЩЕГО ДИАЛОГА:

{dialog_context}


ТЕКУЩЕЕ УКАЗАНИЕ ПОЛЬЗОВАТЕЛЯ:

{user_instruction}


Подготовь ровно три варианта ответа.

ВАЖНЫЕ ПРАВИЛА:

1. Пиши так, как обычно пишет пользователь.

2. Не исправляй намеренно его характерную
манеру английского языка.

3. Не придумывай факты, события,
намерения или чувства, которых нет
в предоставленном контексте.

4. Последние сообщения текущего диалога
имеют больший приоритет, чем старая память.

5. Исторические примеры показывают,
как пользователь действительно отвечал
в похожих ситуациях.

6. Не копируй исторические ответы
механически. Используй их как примеры
стиля, реакции и поведения.

7. Учитывай, что старые исторические
факты могли измениться.

8. Если исторический пример противоречит
текущему диалогу, доверяй текущему
диалогу.

9. Каждый вариант может состоять
из одного или нескольких сообщений,
если пользователь обычно отвечал
несколькими сообщениями.

10. Не объясняй свой выбор.
Верни только три варианта ответа.
""".strip()
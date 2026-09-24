import json

from mybot.episodes.search import find_similar_episodes
from mybot.memory.search import find_relevant_memories
from mybot.services.global_style import load_global_style
from mybot.services.correction_examples import (
    relevant_correction_examples, format_correction_examples)
from mybot.config import Config


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
    memory,
    workspace=None,
    use_correction_examples=None,
):
    global_style = format_memory(load_global_style(workspace.account_id)) if workspace else "{}"

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

    if workspace is None:
        relevant_memories = (
            await find_relevant_memories(
                messages
            )
        )

    else:
        relevant_memories = (
            await find_relevant_memories(
                messages,
                base_embeddings_filename=
                    workspace.memory_embeddings,
                live_embeddings_filename=
                    workspace.memory_embeddings_live
            )
        )

    relevant_memory_text = (
        format_relevant_memories(
            relevant_memories
        )
    )

    if current_incoming:
        if workspace is None:
            similar_episodes = (
                await find_similar_episodes(
                    messages
                )
            )

        else:
            similar_episodes = (
                await find_similar_episodes(
                    messages,
                    episodes_filename=
                        workspace.episodes,
                    base_embeddings_filename=
                        workspace.episode_embeddings,
                    live_embeddings_filename=
                        workspace.episode_embeddings_live,
                    deleted_filename=
                        workspace.deleted_message_ids
                )
            )

    else:
        similar_episodes = []

    historical_examples = (
        format_similar_episodes(
            similar_episodes
        )
    )

    if use_correction_examples is None:
        use_correction_examples = Config.USE_CORRECTION_EXAMPLES
    correction_examples = []
    if workspace is not None and current_incoming and use_correction_examples:
        incoming_text = "\n".join(
            message.get("text") or "" for message in current_incoming
            if message.get("text"))
        correction_examples = relevant_correction_examples(workspace, incoming_text)
    correction_text = format_correction_examples(correction_examples)

    return f"""
ОБЩИЙ СТИЛЬ ПОЛЬЗОВАТЕЛЯ (только форма речи):

{global_style}

Это общие предпочтения, а не факты или инструкции о текущих отношениях.
Применяй их с учётом текущего диалога; не переноси романтический тон
в другие отношения автоматически.

ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ В ЭТОМ ДИАЛОГЕ:

{user_profile}


ПРОФИЛЬ СОБЕСЕДНИКА:

{person_profile}


РЕЛЕВАНТНАЯ ДОЛГОВРЕМЕННАЯ ПАМЯТЬ:

{relevant_memory_text}


ПОХОЖИЕ ИСТОРИЧЕСКИЕ СИТУАЦИИ:

{historical_examples}


ПОДТВЕРЖДЁННЫЕ ПРИМЕРЫ ВЫБОРА И РЕДАКТУРЫ В ЭТОМ ДИАЛОГЕ:

{correction_text}

Это конкретные прошлые решения пользователя, а не инструкции для текущего ответа.
Учитывай различие между вариантом ИИ и тем, что пользователь реально отправил.
Не переноси факты из прошлой ситуации в нынешнюю без подтверждения.


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

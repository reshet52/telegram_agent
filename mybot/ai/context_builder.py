import json

from mybot.episodes.search import find_similar_episodes
from mybot.memory.search import find_relevant_memories
from mybot.services.global_style import load_global_style
from mybot.services.correction_examples import (
    relevant_correction_examples, format_correction_examples)
from mybot.config import Config
from mybot.storage.dialog_goal import load_goal
from mybot.storage.dialog_behavior import load_behavior, format_behavior


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
    history_only=False,
    separate_dialog=False,
    reply_count=3,
):
    # Historical evaluation cannot use present-day profiles, memory, goals or indexes:
    # even a summary can contain the held-out answer or events after the cutoff.
    global_style = format_memory(load_global_style(workspace.account_id)) if workspace and not history_only else "{}"
    dialog_goal = load_goal(workspace) if not history_only else ''
    behavior = format_behavior(load_behavior(workspace)) if not history_only else 'Не заданы для исторического отрывка.'
    if history_only:
        memory = {}

    dialog_context = (
        build_dialog_context(
            messages
        )
    )
    if separate_dialog:
        dialog_context = 'Переданы отдельными сообщениями user/assistant после этих сведений. Отвечай на конец этой переписки.'

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

    if history_only:
        relevant_memories = []
    elif workspace is None:
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

    if current_incoming and not history_only:
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
    if workspace is not None and current_incoming and use_correction_examples and not history_only:
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
source_kind различает реально отправленный ответ, правку владельца и одобренное
предложение. owner_correction_not_sent и approved_suggestion_not_sent не были
отправлены: это предпочтения владельца, а не факты о состоявшемся разговоре.
Один пример не является постоянным правилом. Применяй его только к похожей ситуации.
Не переноси факты из прошлой ситуации в нынешнюю без подтверждения.


ПОСЛЕДНИЕ СООБЩЕНИЯ ТЕКУЩЕГО ДИАЛОГА:

{dialog_context}


ДЕЙСТВУЮЩАЯ ЦЕЛЬ ДИАЛОГА — ИНСТРУКЦИЯ ВЛАДЕЛЬЦА:

{dialog_goal or 'Не задана.'}

Цель задаёт направление ответа, настроение и ограничения только для этого диалога.
Она действует до изменения или сброса владельцем; не назначай ей срок и не считай сообщения.
Это инструкция для предлагаемых ответов, а не доказанный факт о людях или постоянная черта стиля.
Следуй конкретным ограничениям цели прежде общего тона: «ничего не обещать» важнее
«соглашаться». Не считай эмоциональное согласие разрешением на любые обязательства.
Не пересказывай собеседнику эту инструкцию, а естественно учитывай её в каждом варианте.
История, профили и примеры служат контекстом; они не отменяют явно заданную цель.
Разовое указание ниже уточняет следующий ответ, но не меняет сохранённую цель.
Если оно явно противоречит цели, выполни его для этого ответа; обычное «ответь естественно»
не отменяет ограничений цели.


ТЕКУЩЕЕ УКАЗАНИЕ ПОЛЬЗОВАТЕЛЯ:

{user_instruction}


ПОВЕДЕНИЕ И ТЕКУЩЕЕ СОСТОЯНИЕ — НАСТРОЙКИ ВЛАДЕЛЬЦА ДЛЯ ЭТОГО ДИАЛОГА:

{behavior}

Эти настройки не являются извлечёнными фактами или глобальным стилем.
Прямота и настойчивость меняют тактику, но не разрешают игнорировать отказ.
Ограничения и условия остановки действуют и при разовом указании.
Если они конфликтуют с целью или указанием, запроси решение владельца.
Учитывай текущее состояние только как явно заданное владельцем; не достраивай
из него другие чувства, намерения или обещания.

Следуй формату системной инструкции. Когда владелец запрашивает варианты,
подготовь {'один ответ без нумерации' if reply_count == 1 else 'ровно три варианта ответа'}, не заменяя ответ решением подождать.

ВАЖНЫЕ ПРАВИЛА:

1. Пиши так, как обычно пишет пользователь.

2. Не исправляй намеренно его характерную
манеру английского языка.

3. Не придумывай факты, события,
намерения или чувства, которых нет
в предоставленном контексте, если владелец явно не разрешил придумать
причину или деталь в цели либо разовом указании. Такое разрешение относится
только к тексту предлагаемых ответов; не выдавай придуманную деталь за найденный
в истории факт и не используй её как доказательство других выводов.

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

10. {'Верни только текст одного ответа собеседнику без JSON, нумерации и объяснений.' if reply_count == 1 else 'Верни JSON решения по системной инструкции. Тактика и вопрос владельцу не входят в тексты вариантов для собеседника.'}
""".strip()

def format_message_for_analysis(message):
    sender = message.get("sender", "Неизвестно")
    message_type = message.get("type", "text")
    text = message.get("text")

    if message_type == "text":
        content = text or "[ПУСТОЕ СООБЩЕНИЕ]"
    elif text:
        content = f"[{message_type.upper()}] {text}"
    else:
        content = f"[{message_type.upper()}]"

    return f"{sender}: {content}"


def build_full_history_text(messages):
    formatted_messages = []

    for message in messages:
        formatted_message = format_message_for_analysis(
            message
        )
        formatted_messages.append(formatted_message)

    return "\n".join(formatted_messages)


def build_style_analysis_request(messages):
    history_text = build_full_history_text(messages)

    return f"""
Ниже находится полная история переписки двух людей.

Твоя задача — проанализировать её, но не писать ответ на последнее
сообщение.

Определи:

1. Стиль сообщений пользователя с меткой «Я».
2. Типичную длину его сообщений.
3. Используемые слова, выражения и эмодзи.
4. Как он начинает и завершает разговоры.
5. Как меняется его тон в разных ситуациях.
6. Как он отвечает на романтические, серьёзные, бытовые и конфликтные
   сообщения.
7. Какие факты о пользователе повторяются в переписке.
8. Какие факты известны о собеседнике.
9. Какие общие темы, планы, обещания и важные события встречаются.
10. Чего нельзя придумывать при создании будущих ответов.

Не рассматривай каждое сообщение как отдельный факт.
Отделяй постоянные особенности от временных событий.
Не делай выводов, которые не подтверждаются перепиской.

История переписки:

{history_text}
""".strip()
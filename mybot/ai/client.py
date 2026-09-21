from openai import AsyncOpenAI

from mybot.config import Config


ai_client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY
)


async def test_ai_connection():
    response = await ai_client.responses.create(
        model=Config.OPENAI_MODEL,
        input="Ответь одним словом: работает?"
    )

    return response.output_text


async def generate_answers(ai_request):
    response = await ai_client.responses.create(
        model=Config.OPENAI_MODEL,

        instructions=(
            "Ты составляешь ответы для переписки "
            "в Telegram. "
            "Используй предоставленный текущий диалог, "
            "релевантную долговременную память и "
            "исторические примеры реальных ответов "
            "пользователя. "
            "Текущий диалог и прямое указание пользователя "
            "имеют наивысший приоритет. "
            "Верни ровно три разных варианта ответа. "
            "Не добавляй анализ, вступление или пояснения."
        ),

        input=ai_request,

        max_output_tokens=1000,

        store=False
    )

    print("\nИспользование финальной модели:")
    print(
        f"Input tokens: "
        f"{response.usage.input_tokens}"
    )
    print(
        f"Output tokens: "
        f"{response.usage.output_tokens}"
    )
    print(
        f"Total tokens: "
        f"{response.usage.total_tokens}"
    )

    return response.output_text


async def analyze_full_history(history_part):
    response = await ai_client.responses.create(
        model=Config.ANALYSIS_MODEL,

        instructions="""
Ты извлекаешь компактную долговременную память
из части Telegram-переписки.

Сообщения пользователя обозначены как "Я".

Это НЕ литературный анализ.
Не пересказывай всю переписку.
Не пиши длинное эссе.
Не повторяй одну информацию разными словами.

Твоя задача — сохранить только информацию,
которая может помочь другому ИИ в будущем
отвечать так, как реально отвечал пользователь.

Особенно важны:

1. Устойчивые особенности стиля пользователя.
2. Повторяющиеся реакции на конкретные ситуации.
3. Важные факты о пользователе.
4. Важные факты о собеседнике.
5. Важные факты об их отношениях и истории общения.
6. Изменения поведения или отношений.
7. Необычные или очень характерные реакции пользователя.

ПРАВИЛА:

- Не придумывай отсутствующие факты.
- Не анализируй психологию пользователя.
- Не делай выводов о скрытых мотивах.
- Не исправляй английский пользователя.
- Не превращай единичный случай в устойчивую привычку.
- Не сохраняй банальные детали, которые бесполезны
  для будущего ответа.
- Не копируй длинные сообщения целиком.
- Будь максимально компактным.
""",

        input=f"""
Проанализируй эту часть переписки.

Верни ТОЛЬКО JSON следующей структуры:

{{
  "style_patterns": [
    {{
      "pattern": "краткое описание",
      "confidence": "high/medium/low"
    }}
  ],

  "behavior_patterns": [
    {{
      "situation": "что происходит",
      "reaction": "как обычно реагирует пользователь",
      "confidence": "high/medium/low"
    }}
  ],

  "user_facts": [
    {{
      "fact": "факт",
      "confidence": "high/medium/low"
    }}
  ],

  "person_facts": [
    {{
      "fact": "факт",
      "confidence": "high/medium/low"
    }}
  ],

  "relationship_facts": [
    {{
      "fact": "факт",
      "confidence": "high/medium/low"
    }}
  ],

  "important_events": [
    {{
      "event": "краткое описание события"
    }}
  ]
}}

Не добавляй текст до или после JSON.

ПЕРЕПИСКА:

{history_part}
"""
    )

    print("\nИспользование API:")
    print(f"Input tokens: {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")
    print(f"Total tokens: {response.usage.total_tokens}")

    return response.output_text


async def merge_memory_category(
    category,
    items
):
    import json

    items_json = json.dumps(
        items,
        ensure_ascii=False,
        indent=2
    )

    response = await ai_client.responses.create(
        model=Config.ANALYSIS_MODEL,

        instructions="""
Ты создаёшь долговременную память
Telegram AI-агента на основе анализа
реальной истории переписки.

Тебе передаётся одна категория наблюдений,
собранных из последовательных частей истории.

Твоя задача:

- объединить дубликаты;
- объединить очень похожие наблюдения;
- сохранить существенные различия;
- не придумывать новые факты;
- убрать очевидные слабые предположения;
- не превращать наблюдения в психологические диагнозы;
- учитывать, что поздние части истории происходят
  позже ранних частей;
- если что-то изменилось со временем,
  не уничтожать старую информацию,
  а явно сохранить изменение;
- не исправлять стиль английского пользователя;
- сохранить только информацию,
  полезную для будущего составления ответов.

ВАЖНО:
source_chunk показывает положение наблюдения
в истории.

Например:
chunk_002 произошёл раньше chunk_030.

Если раннее и позднее наблюдения различаются,
это может быть изменением во времени,
а не ошибкой.
""",

        input=f"""
КАТЕГОРИЯ:
{category}

НАБЛЮДЕНИЯ:
{items_json}

Верни ТОЛЬКО JSON-массив.

Каждый элемент должен иметь структуру:

{{
    "memory": "краткая информация",
    "confidence": "high/medium/low",
    "evidence": "repeated/single",
    "time_context": "stable/early/later/changed"
}}

Не добавляй никакого текста до или после JSON.
"""
    )

    print(
        f"{category}: "
        f"input={response.usage.input_tokens}, "
        f"output={response.usage.output_tokens}"
    )

    return response.output_text


async def analyze_incremental_memory(
    history_part
):
    response = await ai_client.responses.create(
        model=Config.ANALYSIS_MODEL,

        instructions="""
Ты обновляешь долговременную память
Telegram AI-агента.

Тебе передаётся только НОВАЯ часть
реальной переписки, которой раньше
не было в долговременной памяти.

Сообщения пользователя обозначены как "Я".

Сохраняй только информацию, которая
будет полезна для будущих ответов.

Категории:

- style_patterns
- behavior_patterns
- user_facts
- person_facts
- relationship_facts
- important_events

ВАЖНО:

- не пересказывай переписку;
- не сохраняй банальные сообщения;
- не сохраняй каждое "I love you";
- не превращай один случай в привычку;
- сохраняй новые важные факты;
- сохраняй заметные изменения;
- сохраняй новые характерные реакции;
- сохраняй новые особенности стиля,
  если они действительно проявились;
- не придумывай скрытые мотивы;
- не ставь психологические диагнозы;
- не исправляй английский пользователя.

Верни только JSON-массив.
""",

        input=f"""
НОВАЯ ЧАСТЬ ПЕРЕПИСКИ:

{history_part}

Верни JSON-массив такого вида:

[
  {{
    "category": "behavior_patterns",
    "memory": "краткая полезная информация",
    "confidence": "high/medium/low",
    "evidence": "repeated/single",
    "time_context": "later"
  }}
]

Если в новой переписке нет ничего,
что стоит сохранять в долговременную
память, верни:

[]
"""
    )

    return response.output_text
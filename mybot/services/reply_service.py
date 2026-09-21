import asyncio
import re

from mybot.ai.client import generate_answers
from mybot.ai.context_builder import (
    build_ai_request
)
from mybot.config import Config


DEFAULT_INSTRUCTION = (
    "Ответь естественно, "
    "полностью сохраняя "
    "мой стиль общения."
)


class ReplyService:
    def __init__(
        self,
        recent_messages,
        memory
    ):
        self.recent_messages = (
            recent_messages
        )

        self.memory = memory

        self.generation_lock = (
            asyncio.Lock()
        )


    def apply_mood(
        self,
        instruction,
        current_mood
    ):
        if not current_mood:
            return instruction

        return f"""
{instruction}

ВРЕМЕННОЕ СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЯ:

{current_mood}

Это состояние относится только к текущей
сессии и не является постоянным фактом
о пользователе.

Учитывай его при составлении ответа.

ВАЖНО:
- не сообщай состояние напрямую,
  если пользователь сам этого не попросил;
- передавай его через тон, энергию,
  длину, эмоциональность и формулировки;
- сохраняй естественную реакцию
  на сообщения собеседника;
- сохраняй обычный стиль пользователя;
- любовный ответ может оставаться любовным,
  но с указанным эмоциональным оттенком.
""".strip()


    async def generate(
        self,
        instruction=None,
        current_mood=None
    ):
        if not instruction:
            instruction = (
                DEFAULT_INSTRUCTION
            )

        instruction = self.apply_mood(
            instruction,
            current_mood
        )

        messages_snapshot = list(
            self.recent_messages
        )

        async with self.generation_lock:
            ai_request = (
                await build_ai_request(
                    messages_snapshot,
                    instruction,
                    self.memory
                )
            )

            with open(
                Config.AI_REQUEST_PREVIEW,
                "w",
                encoding="utf-8"
            ) as file:
                file.write(
                    ai_request
                )

            answers = (
                await generate_answers(
                    ai_request
                )
            )

        return answers


    @staticmethod
    def split_variants(text):
        pattern = (
            r"(?ms)^\s*[123][\.\)]\s*"
            r"(.*?)"
            r"(?=^\s*[123][\.\)]\s*|\Z)"
        )

        variants = [
            item.strip()
            for item in re.findall(
                pattern,
                text
            )
        ]

        if len(variants) == 3:
            return variants

        return [
            text.strip()
        ]
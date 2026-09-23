import asyncio
import json
import re
from pathlib import Path
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
        memory,
        workspace=None
    ):
        self.recent_messages = (
            recent_messages
        )

        self.memory = memory
        self.workspace = workspace
        self.last_context_message_id = None

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

        self.last_context_message_id = max(
            (item.get("message_id") for item in messages_snapshot
             if item.get("message_id") is not None), default=None)

        async with self.generation_lock:
            ai_request = (
                await build_ai_request(
                    messages_snapshot,
                    instruction,
                    self.memory,
                    workspace=self.workspace
                )
            )

            if self.workspace is None:
                preview_filename = (
                    Config.AI_REQUEST_PREVIEW
                )
            else:
                preview_filename = (
                    self.workspace.ai_request_preview
                )

            preview_path = Path(
                preview_filename
            )

            preview_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(
                preview_path,
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
        try:
            parsed = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
            if isinstance(parsed, dict):
                parsed = parsed.get("variants")
            if isinstance(parsed, list) and len(parsed) == 3 and all(
                    isinstance(item, str) and item.strip() for item in parsed):
                return [item.strip() for item in parsed]
        except (ValueError, TypeError):
            pass
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
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text.strip()) if item.strip()]
        if len(paragraphs) == 3:
            return paragraphs
        lines = [item.strip() for item in text.splitlines() if item.strip()]
        if len(lines) == 3:
            return lines
        return []

import asyncio
import json
import re
from pathlib import Path
from mybot.ai.client import generate_answers
from mybot.ai.context_builder import (
    build_ai_request, get_current_incoming
)
from mybot.config import Config
from mybot.ai.reply_contract import parse_decision
from mybot.ai.reply_provider import request_snapshot
from mybot.storage.atomic import write_text
from mybot.services.style_versions import selected
from mybot.services.message_order import context_snapshot


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
        self.last_trigger_text = ""
        self.last_decision = None
        self.last_request_snapshot = None

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


    async def generate(self, *args, **kwargs):
        from mybot.services.online_learning import request_started, request_finished
        request_started(self.workspace)
        try:
            return await self._generate(*args, **kwargs)
        finally:
            request_finished(self.workspace)

    async def _generate(
        self,
        instruction=None,
        current_mood=None,
        allow_decision=False,
        force_reply=False,
    ):
        if not instruction:
            instruction = (
                DEFAULT_INSTRUCTION
            )

        instruction = self.apply_mood(
            instruction,
            current_mood
        )

        async with self.generation_lock:
            messages_snapshot = context_snapshot(self.recent_messages)
            self.last_decision = None
            self.last_request_snapshot = None
            self.last_context_message_id = max(
                (item.get("message_id") for item in messages_snapshot
                 if item.get("message_id") is not None), default=None)
            self.last_trigger_text = "\n".join(
                item.get("text") or "" for item in get_current_incoming(messages_snapshot)
                if item.get("text"))[:2000]
            version = selected(self.workspace) if self.workspace else 'configured'
            if version != 'configured':
                from mybot.ai.local_context import build_local_context
                ai_request = build_local_context(messages_snapshot, instruction, self.memory, self.workspace)
            else:
                ai_request = await build_ai_request(
                    messages_snapshot,
                    instruction,
                    self.memory,
                    workspace=self.workspace,
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

            if version != 'configured':
                from mybot.ai import local_reply
                snapshot = local_reply.snapshot(ai_request, self.workspace, version,
                    messages=local_reply.live_messages(ai_request, messages_snapshot))
            else:
                snapshot = request_snapshot(ai_request, allow_decision=allow_decision)
            snapshot['context_message_id'] = self.last_context_message_id
            snapshot['context_message_ids'] = [m['message_id'] for m in messages_snapshot
                                                if m.get('message_id') is not None]
            write_text(preview_path, json.dumps(snapshot, ensure_ascii=False, indent=2))

            if version != 'configured':
                if allow_decision and not force_reply:
                    from mybot.services.decision_task import messages as decision_messages
                    decision_request = local_reply.snapshot('', self.workspace, version,
                        messages=decision_messages(messages_snapshot), count=1)
                    decision_request['context_message_ids'] = snapshot['context_message_ids']
                    try:
                        action = await local_reply.decide(decision_request)
                    finally:
                        snapshot['decision_request'] = decision_request
                        write_text(preview_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
                    if action == 'wait':
                        self.require_current(messages_snapshot)
                        answers = json.dumps({'action': 'wait', 'tactic': 'Пауза',
                            'owner_note': 'Сейчас лучше не отвечать. Если нужен ответ, запросите варианты явно.',
                            'variants': []}, ensure_ascii=False)
                        self.last_decision = parse_decision(answers)
                        self.last_request_snapshot = snapshot
                        return answers
                answers = json.dumps({'variants': await local_reply.generate(snapshot)}, ensure_ascii=False)
                write_text(preview_path, json.dumps(snapshot, ensure_ascii=False, indent=2))
            else:
                answers = await generate_answers(ai_request, allow_decision=allow_decision)
            self.require_current(messages_snapshot)
            self.last_decision = parse_decision(answers)
            if not allow_decision and self.last_decision and self.last_decision['action'] != 'reply':
                self.last_decision = None
                raise ValueError('Модель не вернула запрошенные варианты ответа. Повторите запрос.')
            self.last_request_snapshot = snapshot

        return answers

    def require_current(self, snapshot):
        if context_snapshot(self.recent_messages) != snapshot:
            raise ValueError('Переписка изменилась во время генерации. Нажмите «Ответ» ещё раз, '
                             'чтобы получить варианты для последних сообщений.')


    @staticmethod
    def split_variants(text):
        decision = parse_decision(text)
        if decision is not None:
            return decision['variants']
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

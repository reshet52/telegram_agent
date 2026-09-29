"""Offline tests: synthetic history and mocked model, no account/API access."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.app.session_state import SessionState
from mybot.services.random_fact import RandomFact, RandomFactService, sample_history, validate_fact
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.bot_keyboards import main_menu


class RandomFactTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.workspace(1, 42)
        self.records = [self.message(1, "I live in Toronto."),
                        self.message(2, "Nice!", sender="Я")]
        self.save(self.ws, self.records)

    def workspace(self, account, dialog):
        ws = Workspace(account, dialog, "Anna", self.root / f"account_{account}" / f"dialog_{dialog}")
        ws.root.mkdir(parents=True)
        return ws

    def message(self, mid, text, sender="Anna"):
        return {"message_id": mid, "text": text, "sender": sender,
                "date": "2024-01-01T12:00:00+00:00", "type": "text"}

    def save(self, ws, records):
        ws.chat_history.write_text("\n\n".join(json.dumps(item, indent=2) for item in records))

    def output(self):
        return json.dumps({"fact": "В январе 2024 года Анна писала, что живёт в Торонто.",
                           "sources": [{"message_id": 1, "quote": "I live in Toronto."}]})

    def test_sampling_limits_and_chronological_windows(self):
        self.save(self.ws, [self.message(i, "x" * 2000) for i in range(1, 501)])
        sections = sample_history(self.ws, {1, 2})
        self.assertLessEqual(len(sections), 6)
        self.assertLessEqual(len(json.dumps(sections, ensure_ascii=False)), 16000)
        ids = []
        for section in sections:
            self.assertLessEqual(len(section), 8)
            part = [item['message_id'] for item in section]
            self.assertEqual(part, sorted(part))
            ids.extend(part)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(1, ids)
        self.assertNotIn(2, ids)

    def test_deleted_media_and_other_workspaces_are_excluded(self):
        self.save(self.ws, [*self.records, self.message(3, None), self.message(4, "deleted")])
        # Use the project's actual deletion writer/reader format.
        from mybot.storage.deletions import mark_messages_deleted
        mark_messages_deleted([4], filename=self.ws.deleted_message_ids)
        for account, dialog in ((2, 42), (1, 43)):
            self.save(self.workspace(account, dialog), [self.message(1, "another workspace secret")])
        evidence = sample_history(self.ws, set())
        self.assertEqual({m['message_id'] for s in evidence for m in s}, {1, 2})
        self.assertNotIn("secret", json.dumps(evidence))

    def test_validated_citation_preserves_original_metadata(self):
        result = validate_fact(self.output(), [self.records], set())
        card = result.format("Anna")
        self.assertIn("I live in Toronto.", card)
        self.assertIn("Anna · 2024-01-01", card)
        self.assertIn("ID 1", card)

    def test_invalid_payloads_and_fabricated_sources_are_rejected(self):
        for raw in ("bad JSON", "[]", "{}", '{"fact": 3}',
                    json.dumps({"fact": "Fact", "sources": [{"message_id": 999, "quote": "I live in Toronto."}]}),
                    json.dumps({"fact": "Fact", "sources": [{"message_id": 1, "quote": "I live in Paris."}]}),
                    json.dumps({"fact": "Fact", "sources": [{"message_id": True, "quote": "I live in Toronto."}]})):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_fact(raw, [self.records], set())
        self.assertIsNone(validate_fact('{"fact":null}', [self.records], set()))

    async def test_one_request_and_recent_sources_excluded_without_writes(self):
        service = RandomFactService()
        before = self.ws.chat_history.read_bytes()
        create = AsyncMock(return_value=NS(output_text=self.output()))
        with patch("mybot.services.random_fact.ai_client.with_options",
                   return_value=NS(responses=NS(create=create))) as options:
            fact = await service.extract(self.ws)
            service.remember(self.ws, fact)
            with self.assertRaises(ValueError):
                await service.extract(self.ws)
        options.assert_called_once_with(max_retries=0)
        create.assert_awaited_once()
        self.assertFalse(create.await_args.kwargs['store'])
        self.assertEqual(before, self.ws.chat_history.read_bytes())
        self.assertEqual(list(self.ws.root.iterdir()), [self.ws.chat_history])

    async def test_empty_history_does_not_call_model_and_api_failure_is_not_retried(self):
        service = RandomFactService()
        with patch("mybot.services.random_fact.ai_client.with_options") as options:
            self.save(self.ws, [])
            with self.assertRaises(ValueError):
                await service.extract(self.ws)
            options.assert_not_called()
        self.save(self.ws, self.records)
        create = AsyncMock(side_effect=RuntimeError("offline API error"))
        with patch("mybot.services.random_fact.ai_client.with_options",
                   return_value=NS(responses=NS(create=create))):
            with self.assertRaises(RuntimeError):
                await service.extract(self.ws)
        create.assert_awaited_once()

    def test_recent_memory_is_bounded_and_scoped(self):
        service = RandomFactService()
        for i in range(25):
            service.remember(self.ws, RandomFact(str(i), [{'message_id': i}]))
        self.assertEqual(len(service.recent[(1, 42)]), 20)
        self.assertEqual(service.recent[(1, 42)][0][0], "5")
        self.assertNotIn((2, 42), service.recent)
        self.assertNotIn((1, 43), service.recent)
        self.assertIsNone(validate_fact(self.output(), [self.records],
            {json.loads(self.output())['fact'].casefold()}))

    async def test_ui_preserves_card_and_rejects_stale_dialog_and_api_error(self):
        bot = BotInterface(1, [], object(), SessionState(), workspace=self.ws)
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.menu = NS(show=AsyncMock())
        message = NS(reply_text=AsyncMock(), edit_text=AsyncMock())
        update = NS(effective_message=message, effective_user=NS(id=1),
                    effective_chat=NS(id=1, type="private"))
        fact = validate_fact(self.output(), [self.records], set())
        bot.random_facts.extract = AsyncMock(return_value=fact)
        await bot.callbacks.dispatch("ui:random_fact", update, NS(user_data={}))
        message.reply_text.assert_awaited_once()
        message.edit_text.assert_not_awaited()
        self.assertEqual(len(bot.random_facts.recent[(1, 42)]), 1)
        await bot.callbacks.dispatch("ui:random_fact:43", update, NS(user_data={}))
        bot.random_facts.extract.assert_awaited_once()
        bot.random_facts.extract = AsyncMock(side_effect=RuntimeError("offline"))
        with self.assertLogs(level="ERROR"):
            await bot.random_fact_command(update)
        self.assertIn("Не удалось получить факт", bot.menu.show.await_args.args[1])
        message.reply_text.assert_awaited_once()
        bot.workspace = None
        await bot.random_fact_command(update)
        bot.random_facts.extract.assert_awaited_once()

    def test_button_only_in_active_menu(self):
        for active in (True, False):
            actions = [button.callback_data for row in main_menu(active).inline_keyboard for button in row]
            self.assertEqual('ui:random_fact' in actions, active)


if __name__ == '__main__':
    unittest.main()

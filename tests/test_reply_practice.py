"""Historical holdout, no future retrieval, persisted ratings and offline UI flows."""

import asyncio
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.ai.reply_contract import VARIANT_INSTRUCTIONS
from mybot.app.session_state import SessionState
from mybot.services.reply_practice import generate_practice
from mybot.services.reply_service import ReplyService
from mybot.storage.atomic import write_json
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_reply_practice import rating_markup


class PracticeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = Workspace(1, 10, 'Test', self.root / 'one')
        self.other = Workspace(2, 10, 'Other', self.root / 'two')
        self.messages = [
            {'message_id': i, 'sender': sender, 'text': text, 'type': 'text', 'date': '2025-01-01'}
            for i, sender, text in [
                (1, 'Я', 'PAST_STYLE'), (2, 'Other', 'VISIBLE_QUESTION'),
                (3, 'Я', 'HIDDEN_TARGET_ONE'), (4, 'Я', 'HIDDEN_TARGET_TWO'),
                (5, 'Other', 'FUTURE_QUESTION'), (6, 'Я', 'FUTURE_REPLY')]]
        self.write_history()
        self.store = PracticeStore(self.ws)
        patcher = patch('mybot.storage.reply_practice.random.choice', side_effect=lambda items: items[0])
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_history(self):
        self.ws.root.mkdir(parents=True, exist_ok=True)
        # Preserve the repository's legacy multiline JSON format.
        self.ws.chat_history.write_text(''.join(json.dumps(m, ensure_ascii=False, indent=2) + '\n\n'
                                                for m in self.messages))

    def forbidden_retrieval(self, stack):
        for name in ('load_global_style', 'load_goal', 'load_behavior', 'relevant_correction_examples',
                     'find_relevant_memories', 'find_similar_episodes'):
            stack.enter_context(patch('mybot.ai.context_builder.' + name,
                                     side_effect=AssertionError('Future data accessed: ' + name)))

    async def test_holdout_entire_turn_and_no_future_sources_or_duplicate_request(self):
        case = self.store.create()
        self.assertEqual(case['context_ids'], [1, 2])
        self.assertEqual(case['expected_ids'], [3, 4])
        with ExitStack() as stack:
            self.forbidden_retrieval(stack)
            api = stack.enter_context(patch('mybot.services.reply_practice.generate_answers',
                AsyncMock(return_value='{"variants":["one","two","three"]}')))
            result = await generate_practice(self.store, case['id'])
            again = await generate_practice(self.store, case['id'])
        api.assert_awaited_once()
        prompt = api.await_args.args[0]
        self.assertIn('VISIBLE_QUESTION', prompt)
        self.assertIn('PAST_STYLE', prompt)
        for marker in ('HIDDEN_TARGET_ONE', 'HIDDEN_TARGET_TWO', 'FUTURE_QUESTION', 'FUTURE_REPLY'):
            self.assertNotIn(marker, prompt)
            self.assertNotIn(marker, json.dumps(result['request']))
        self.assertEqual(result, again)
        self.assertEqual(result['request']['instructions'], VARIANT_INSTRUCTIONS)
        self.assertFalse(self.ws.reply_journal.exists())
        self.assertFalse(self.ws.ai_request_preview.exists())

    async def test_isolation_restart_ratings_and_reveal(self):
        case = self.store.create()
        key = case['id']
        with self.assertRaises(ValueError):
            self.store.reveal(key)
        self.store.save_result(key, ['a', 'b', 'c'])
        self.store.rate(key, 1, 'good')
        restarted = PracticeStore(self.ws)
        self.assertFalse(restarted.get(key)['ratings']['1']['after_reveal'])
        restarted.reveal(key)
        restarted.rate(key, 1, 'style')
        self.assertTrue(restarted.get(key)['ratings']['1']['after_reveal'])
        self.assertEqual(restarted.get(key)['ratings']['1']['rating'], 'style')
        self.assertEqual(len(restarted.get(key)['ratings']), 1)
        for wrong in [self.other, Workspace(2, 10, 'Wrong', self.ws.root)]:
            with self.assertRaises(ValueError):
                PracticeStore(wrong).rate(key, 1, 'good')
        with self.assertRaises(ValueError):
            self.store.get('../elsewhere')

    async def test_deleted_or_edited_sources_block_generation_and_reveal(self):
        case = self.store.create()
        self.store.save_result(case['id'], ['a', 'b', 'c'])
        mark_messages_deleted({3}, filename=self.ws.deleted_message_ids)
        with self.assertRaises(ValueError):
            self.store.reveal(case['id'])
        with self.assertRaises(ValueError):
            self.store.rate(case['id'], 1, 'good')
        with self.assertRaises(ValueError):
            await generate_practice(self.store, case['id'])
        # Deletion inside a candidate's context excludes the entire candidate.
        with self.assertRaises(ValueError):
            self.store.create()

    async def test_edit_after_request_prevents_saving_stale_result(self):
        case = self.store.create()
        async def api(prompt):
            self.messages[1]['text'] = 'EDITED_QUESTION'
            self.write_history()
            return '{"variants":["a","b","c"]}'
        with ExitStack() as stack:
            self.forbidden_retrieval(stack)
            stack.enter_context(patch('mybot.services.reply_practice.generate_answers', side_effect=api))
            with self.assertRaises(ValueError):
                await generate_practice(self.store, case['id'])
        self.assertEqual(self.store.get(case['id'])['variants'], [])

    async def test_retries_are_explicit_and_invalid_response_not_recorded_as_success(self):
        case = self.store.create()
        with ExitStack() as stack:
            self.forbidden_retrieval(stack)
            api = stack.enter_context(patch('mybot.services.reply_practice.generate_answers',
                                            AsyncMock(return_value='no variants')))
            with self.assertRaises(ValueError):
                await generate_practice(self.store, case['id'])
        api.assert_awaited_once()
        self.assertEqual(self.store.get(case['id'])['variants'], [])
        self.assertIn('request', self.store.get(case['id']))

    async def test_sampling_does_not_repeat_and_skips_media(self):
        first = self.store.create()
        second = self.store.create()
        self.assertNotEqual(first['expected_ids'][0], second['expected_ids'][0])
        with self.assertRaises(ValueError):
            self.store.create()
        self.messages[1]['type'] = 'photo'
        self.write_history()
        # New store without previous selections but same source.
        fresh = Workspace(1, 11, 'Fresh', self.root / 'fresh')
        fresh.root.mkdir()
        fresh.chat_history.write_text(self.ws.chat_history.read_text())
        with self.assertRaises(ValueError):
            PracticeStore(fresh).create()

    async def test_atomic_save_failure_and_corrupt_cutoff(self):
        case = self.store.create()
        self.store.save_result(case['id'], ['a', 'b', 'c'])
        with patch('mybot.storage.atomic.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                self.store.rate(case['id'], 1, 'good')
        self.assertEqual(self.store.get(case['id'])['ratings'], {})
        case['context_ids'] = [1, 2, 5]
        write_json(self.store.path(case['id']), case)
        with self.assertRaises(ValueError):
            self.store.get(case['id'])

    async def test_ui_preview_run_rate_reveal_and_stale_dialog(self):
        bot = BotInterface(1, [], None, SessionState(), self.ws, telegram_client=Mock())
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.callbacks.show = AsyncMock()
        bot.send_long_text = AsyncMock()
        update = NS(effective_message=NS(edit_reply_markup=AsyncMock()), effective_user=NS(id=1),
                    effective_chat=NS(id=1, type='private'))
        context = NS(user_data={})
        with ExitStack() as stack:
            self.forbidden_retrieval(stack)
            api = stack.enter_context(patch('mybot.services.reply_practice.generate_answers',
                                            AsyncMock(return_value='{"variants":["a","b","c"]}')))
            await bot.reply_practice.handle('ui:rp:new:1:10', update, context)
            api.assert_not_awaited()
            key = self.store.records()[0]['id']
            await bot.reply_practice.handle(f'ui:rp:run:{key}', update, context)
            await bot.reply_practice.handle(f'ui:rp:run:{key}', update, context)
            api.assert_awaited_once()
        output = '\n'.join(call.args[1] for call in bot.send_long_text.await_args_list)
        self.assertNotIn('HIDDEN_TARGET_ONE', output)
        await bot.reply_practice.handle(f'ui:rp:rate:{key}:1:good', update, context)
        self.assertEqual(self.store.get(key)['ratings']['1']['rating'], 'good')
        await bot.reply_practice.handle(f'ui:rp:reveal:{key}', update, context)
        self.assertIn('HIDDEN_TARGET_ONE', bot.send_long_text.await_args.args[1])
        self.assertIn('HIDDEN_TARGET_TWO', bot.send_long_text.await_args.args[1])
        bot.workspace = self.other
        await bot.reply_practice.handle('ui:rp:new:1:10', update, context)
        self.assertEqual(PracticeStore(self.other).records(), [])
        await bot.reply_practice.handle(f'ui:rp:rate:{key}:1:facts', update, context)
        self.assertEqual(self.store.get(key)['ratings']['1']['rating'], 'good')
        bot.telegram_client.send_message.assert_not_called()

    async def test_unauthorized_and_busy_ui_do_not_start_work(self):
        bot = BotInterface(1, [], None, SessionState(), self.ws, telegram_client=Mock())
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.callbacks.show = AsyncMock()
        update = NS(effective_message=Mock(), effective_user=NS(id=2),
                    effective_chat=NS(id=2, type='private'))
        await bot.reply_practice.handle('ui:rp:new:1:10', update, NS(user_data={}))
        self.assertEqual(self.store.records(), [])
        update.effective_user.id = update.effective_chat.id = 1
        bot.runtime_controller.busy = True
        await bot.reply_practice.handle('ui:rp:new:1:10', update, NS(user_data={}))
        self.assertEqual(self.store.records(), [])

    async def test_rating_cards_survive_menu_cleanup_and_callback_limits(self):
        case = self.store.create()
        markup = rating_markup(case, 1)
        self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=markup)))
        self.assertTrue(all(len(b.callback_data.encode()) <= 64 for row in markup.inline_keyboard for b in row))

    async def test_manual_reply_requests_variants_and_never_silently_waits(self):
        service = ReplyService(self.messages[:2], {}, self.ws)
        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='context')), \
             patch('mybot.services.reply_service.generate_answers',
                   AsyncMock(return_value='{"variants":["a","b","c"]}')) as api:
            raw = await service.generate()
            self.assertEqual(service.split_variants(raw), ['a', 'b', 'c'])
            self.assertFalse(api.await_args.kwargs['allow_decision'])
            self.assertEqual(service.last_request_snapshot['instructions'], VARIANT_INSTRUCTIONS)
            api.return_value = '{"action":"wait","tactic":"Pause","owner_note":"Wait","variants":[]}'
            with self.assertRaises(ValueError):
                await service.generate()
        self.assertIsNone(service.last_decision)

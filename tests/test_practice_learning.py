"""Approved practice evidence improves live context without contaminating holdouts."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from telegram.error import BadRequest

from mybot.ai.context_builder import build_ai_request
from mybot.services.correction_examples import relevant_correction_examples
from mybot.services.practice_learning import confirmed_examples
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_reply_practice import BotReplyPractice
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.practice_cards import render_case


class PracticeLearningTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = Workspace(1, 10, 'Test', self.root / 'one')
        self.other = Workspace(2, 10, 'Other', self.root / 'two')
        self.ws.root.mkdir()
        self.messages = [dict(message_id=1, sender='Other', text='How was the beach?', type='text'),
                         dict(message_id=2, sender='Я', text='REAL_REPLY', type='text')]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))
        self.store = PracticeStore(self.ws)
        self.case = self.store.create()
        self.key = self.case['id']
        self.store.save_result(self.key, ['APPROVED_REPLY', 'second', 'third'])

    def examples(self):
        return relevant_correction_examples(self.ws, 'How was the beach?')

    async def test_positive_feedback_used_and_reversal_removes_it(self):
        self.assertEqual(self.examples(), [])
        self.store.rate(self.key, 1, 'good')
        item = self.examples()[0]
        self.assertEqual(item['actual_reply'], 'APPROVED_REPLY')
        self.assertEqual(item['source_kind'], 'approved_suggestion_not_sent')
        self.assertEqual(confirmed_examples(self.other), [])
        self.store.rate(self.key, 1, 'facts')
        self.assertEqual(self.examples(), [])

    async def test_original_requires_explicit_choice_and_correction_is_not_a_sent_reply(self):
        self.store.reveal(self.key)
        self.assertEqual(self.examples(), [])
        self.store.use_original(self.key, True)
        self.assertEqual(self.examples()[0]['actual_reply'], 'REAL_REPLY')
        self.assertEqual(self.examples()[0]['source_kind'], 'historical_sent_reply')
        self.store.save_correction(self.key, 'MY_BETTER_REPLY')
        self.assertEqual(self.examples()[0]['actual_reply'], 'MY_BETTER_REPLY')
        self.assertEqual(self.examples()[0]['source_kind'], 'owner_correction_not_sent')
        self.store.clear_learning(self.key)
        self.assertEqual(self.examples(), [])
        self.store.save_correction(self.key, 'REENABLED_REPLY')
        self.assertEqual(self.examples()[0]['actual_reply'], 'REENABLED_REPLY')

    async def test_deleted_source_and_wrong_account_cannot_contribute(self):
        self.store.rate(self.key, 1, 'good')
        wrong = Workspace(2, 10, 'Other', self.ws.root)
        with self.assertRaises(ValueError):
            confirmed_examples(wrong)
        mark_messages_deleted({2}, filename=self.ws.deleted_message_ids)
        self.assertEqual(self.examples(), [])

    async def test_live_prompt_uses_example_but_blind_practice_cannot(self):
        self.store.rate(self.key, 1, 'good')
        incoming = [self.messages[0]]
        with patch('mybot.ai.context_builder.load_global_style', return_value={}), \
             patch('mybot.ai.context_builder.find_relevant_memories', AsyncMock(return_value=[])), \
             patch('mybot.ai.context_builder.find_similar_episodes', AsyncMock(return_value=[])):
            live = await build_ai_request(incoming, 'reply', {}, workspace=self.ws)
            blind = await build_ai_request(incoming, 'reply', {}, workspace=self.ws, history_only=True)
        self.assertIn('APPROVED_REPLY', live)
        self.assertIn('approved_suggestion_not_sent', live)
        self.assertNotIn('APPROVED_REPLY', blind)
        self.assertNotIn('REAL_REPLY', blind)

    async def test_card_is_bounded_and_context_original_paginate(self):
        case = self.store.get(self.key)
        case['variants'][0] = '😀' * 5000
        context, expected = self.store.source(case)
        for view in ('view', 'context', 'original'):
            case['revealed_at'] = 'today'
            text, markup = render_case(self.ws, case, context, expected, view=view)
            self.assertLessEqual(len(text.encode('utf-16-le')) // 2, 4096)
            self.assertTrue(all(len(b.callback_data.encode()) <= 64 for row in markup.inline_keyboard for b in row))
            self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=markup)))
        case['revealed_at'] = None
        with self.assertRaises(ValueError):
            render_case(self.ws, case, context, expected, view='original')
        text, _ = render_case(self.ws, case, context, expected)
        self.assertNotIn('REAL_REPLY', text)
        self.assertNotIn('second', text)

    def bot(self):
        return NS(is_owner=lambda update: True, goal=NS(workspace=lambda: self.ws),
                  runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()),
                  send_long_text=AsyncMock())

    async def test_navigation_edits_one_card_instead_of_sending_another(self):
        bot = self.bot()
        case = self.store.get(self.key)
        context, expected = self.store.source(case)
        _, markup = render_case(self.ws, case, context, expected)
        message = NS(reply_markup=markup, edit_text=AsyncMock())
        update = NS(effective_message=message, callback_query=NS(data=f'ui:rp:view:{self.key}:2:0'))
        ui = BotReplyPractice(bot)
        await ui.handle(update.callback_query.data, update, NS(user_data={}))
        message.edit_text.assert_awaited_once()
        self.assertIn('Вариант 2/3', message.edit_text.await_args.args[0])
        bot.send_long_text.assert_not_awaited()
        message.edit_text.side_effect = BadRequest('Message is not modified')
        await ui.handle(update.callback_query.data, update, NS(user_data={}))
        bot.send_long_text.assert_not_awaited()

    async def test_correction_input_is_saved_and_stale_workspace_is_rejected(self):
        bot = self.bot()
        ui = BotReplyPractice(bot)
        update = NS(effective_message=NS(reply_markup=None), callback_query=None)
        context = NS(user_data={})
        await ui.handle(f'ui:rp:correct:{self.key}', update, context)
        self.assertTrue(await ui.handle_text('CORRECTED_REPLY', update, context))
        self.assertEqual(self.examples()[0]['actual_reply'], 'CORRECTED_REPLY')
        await ui.handle(f'ui:rp:correct:{self.key}', update, context)
        bot.goal.workspace = lambda: self.other
        await ui.handle_text('WRONG_ACCOUNT', update, context)
        self.assertNotIn('practice_correction', context.user_data)
        self.assertEqual(self.store.get(self.key)['owner_correction']['text'], 'CORRECTED_REPLY')

    async def test_new_correction_and_unlearning_are_atomic(self):
        self.store.save_correction(self.key, 'KEEP_ME')
        with patch('mybot.storage.atomic.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                self.store.clear_learning(self.key)
        self.assertEqual(self.examples()[0]['actual_reply'], 'KEEP_ME')
        for text in ('', 'x' * 3501):
            with self.assertRaises(ValueError):
                self.store.save_correction(self.key, text)

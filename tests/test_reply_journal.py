"""Offline records for personalization; messages are never sent here."""
import json
import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import closing
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.storage.workspace import Workspace
from mybot.storage.reply_journal import (
    create_draft, finalize_draft_feedback, link_outgoing_to_draft,
    mark_draft_kept, mark_independent_reply, record_generation,
    record_outgoing_reply, record_outgoing_replies)
from mybot.telegram.live_events import LiveEvents
from mybot.telegram.bot_interface import BotInterface
from mybot.app.session_state import SessionState


class JournalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.ws_a = Workspace(1, 42, 'A', root / 'account_1/dialogs/dialog_42')
        self.ws_b = Workspace(1, 13, 'B', root / 'account_1/dialogs/dialog_13')

    def rows(self, workspace, table):
        with closing(sqlite3.connect(workspace.reply_journal)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(f'SELECT * FROM {table}')]

    async def test_generation_and_real_outgoing_stay_in_separate_workspaces(self):
        generation_id = record_generation(self.ws_a, ['коротко', 'иначе', 'ещё'], 88,
                                          'gpt-6-luna')
        message = {'message_id': 99, 'date': '2026-09-23T10:00:00+00:00',
                   'sender': 'Я', 'text': 'мой настоящий ответ', 'type': 'text'}
        record_outgoing_reply(self.ws_a, message)
        record_outgoing_reply(self.ws_a, message)
        record_outgoing_reply(self.ws_b, message)
        self.assertEqual(len(self.rows(self.ws_a, 'outgoing_replies')), 1)
        self.assertEqual(len(self.rows(self.ws_b, 'outgoing_replies')), 1)
        row = self.rows(self.ws_a, 'outgoing_replies')[0]
        self.assertEqual(row['text'], 'мой настоящий ответ')
        self.assertIsNone(row['generation_id'])
        self.assertIsNone(row['result'])
        saved = self.rows(self.ws_a, 'generations')[0]
        self.assertEqual(saved['generation_id'], generation_id)
        self.assertEqual(saved['context_message_id'], 88)
        self.assertEqual(json.loads(saved['variants_json']), [
            {'variant_id': f'{generation_id}:1', 'text': 'коротко'},
            {'variant_id': f'{generation_id}:2', 'text': 'иначе'},
            {'variant_id': f'{generation_id}:3', 'text': 'ещё'},
        ])
        self.assertEqual(self.rows(self.ws_b, 'generations'), [])

    async def test_incoming_does_not_become_user_reply(self):
        record_outgoing_reply(self.ws_a, {'message_id': 100, 'sender': 'A', 'text': 'hi'})
        self.assertFalse(self.ws_a.reply_journal.exists())

    async def test_restart_backfill_recovers_outgoing_without_duplicates(self):
        messages = [
            {'message_id': 1, 'sender': 'A', 'text': 'question'},
            {'message_id': 2, 'sender': 'Я', 'text': 'answer', 'type': 'text'},
            {'message_id': 3, 'sender': 'Я', 'text': None, 'type': 'photo'},
        ]
        record_outgoing_replies(self.ws_a, messages)
        record_outgoing_replies(self.ws_a, messages)
        self.assertEqual([row['message_id'] for row in self.rows(self.ws_a, 'outgoing_replies')],
                         [2, 3])
        self.assertFalse(self.ws_b.reply_journal.exists())

    async def test_explicit_multimessage_feedback_and_workspace_isolation(self):
        generation_id = record_generation(self.ws_a,
            ['one\n\ntwo', 'another', 'third'], 88, 'gpt-6-luna')
        draft_id = create_draft(self.ws_a, generation_id, 1)
        self.assertTrue(mark_draft_kept(self.ws_a, draft_id))
        for message_id, text in [(101, 'one'), (102, 'two')]:
            record_outgoing_reply(self.ws_a, {'message_id': message_id,
                'sender': 'Я', 'text': text, 'type': 'text'})
        self.assertEqual(link_outgoing_to_draft(self.ws_a, 101, draft_id), 1)
        self.assertEqual(self.rows(self.ws_a, 'outgoing_replies')[0]['result'],
                         'linked_pending')
        self.assertEqual(link_outgoing_to_draft(self.ws_a, 102, draft_id), 2)
        result, actual, count = finalize_draft_feedback(self.ws_a, draft_id)
        self.assertEqual((result, actual, count),
                         ('accepted_without_edit', 'one\n\ntwo', 2))
        self.assertEqual([row['result'] for row in self.rows(self.ws_a, 'outgoing_replies')],
                         ['accepted_without_edit'] * 2)
        with self.assertRaises(ValueError):
            link_outgoing_to_draft(self.ws_b, 101, draft_id)
        self.assertEqual(self.rows(self.ws_b, 'draft_message_links'), [])

    async def test_edited_group_and_independent_reply_are_explicit(self):
        generation_id = record_generation(self.ws_a,
            ['hello', 'other', 'third'], None, 'gpt-6-luna')
        draft_id = create_draft(self.ws_a, generation_id, 1)
        mark_draft_kept(self.ws_a, draft_id)
        for message_id, text in [(201, 'hello again'), (202, 'own words')]:
            record_outgoing_reply(self.ws_a, {'message_id': message_id,
                'sender': 'Я', 'text': text, 'type': 'text'})
        link_outgoing_to_draft(self.ws_a, 201, draft_id)
        self.assertEqual(finalize_draft_feedback(self.ws_a, draft_id)[0],
                         'accepted_with_edit')
        mark_independent_reply(self.ws_a, 202)
        self.assertEqual(self.rows(self.ws_a, 'outgoing_replies')[1]['result'],
                         'independent_reply')
        with self.assertRaises(ValueError):
            link_outgoing_to_draft(self.ws_a, 202, draft_id)
        with self.assertRaises(ValueError):
            mark_independent_reply(self.ws_a, 201)

    async def test_control_bot_records_generated_variants_before_presenting(self):
        bot = BotInterface(1, [{'message_id': 88, 'sender': 'A', 'text': 'hi'}],
                           NS(generate=AsyncMock(return_value='1. a\n2. b\n3. c'),
                              split_variants=lambda value: ['a', 'b', 'c'],
                              last_context_message_id=88),
                           SessionState(), workspace=self.ws_a, telegram_client=Mock())
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.callbacks.show = AsyncMock()
        bot.menu = NS(show=AsyncMock())
        bot.send_long_text = AsyncMock()
        update = NS(effective_message=Mock(), effective_user=NS(id=1),
                    effective_chat=NS(id=1, type='private'))
        await bot.send_generated_answers(update)
        self.assertEqual(len(self.rows(self.ws_a, 'generations')), 1)
        self.assertEqual(self.rows(self.ws_a, 'generations')[0]['context_message_id'], 88)
        self.assertEqual(bot.send_long_text.await_count, 1)
        self.assertIn('Вариант 3\nc', bot.send_long_text.await_args.args[1])

    async def test_live_outgoing_writes_journal_and_history(self):
        listener = LiveEvents(Mock(), NS(id=42), NS(id=1), 'A', self.ws_a,
                              [], [], set(), NS(send_incoming_message=AsyncMock(),
                                                 send_outgoing_message=AsyncMock()), Mock())
        listener.enabled = True
        with patch('mybot.telegram.live_events.message_to_data', new_callable=AsyncMock,
                   return_value={'message_id': 101, 'sender': 'Я', 'text': 'hello',
                                 'type': 'text', 'date': '2026-09-23'}), \
             patch('mybot.telegram.live_events.append_message_data'), \
             patch('mybot.telegram.live_events.process_live_episode_message',
                   new_callable=AsyncMock, return_value=None), patch('builtins.print'):
            await listener.on_new_message(NS(message=Mock()))
        self.assertEqual(self.rows(self.ws_a, 'outgoing_replies')[0]['text'], 'hello')
        listener.bot_interface.send_outgoing_message.assert_awaited_once()

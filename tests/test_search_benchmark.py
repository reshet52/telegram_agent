"""Offline benchmark workflow with fictional fixtures; no live API or private data."""

import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.app.session_state import SessionState
from mybot.services.random_fact import RandomFact
from mybot.services.history_search import search_history
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.search_benchmark import BenchmarkStore
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_benchmark import button
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.bot_menu import BotMenu


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.workspace(1, 42)
        self.store = BenchmarkStore(self.ws)
        self.record = {'message_id': 7, 'sender': 'Test contact', 'date': '2024-01-01',
                       'text': 'My favourite colour is green.'}
        self.ws.chat_history.write_text(json.dumps(self.record, indent=2) + '\n\n')
        self.fact = RandomFact('В тестовом примере любимый цвет — зелёный.',
                               [{**self.record, 'quote': self.record['text']}])
        self.fact_id = self.store.save_fact(self.fact)
        self.draft = {**self.store.fact(self.fact_id), 'question': 'Какой любимый цвет?'}
        self.bot = BotInterface(1, [], object(), SessionState(), workspace=self.ws)
        self.bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        self.bot.menu = NS(show=AsyncMock(), move_to_bottom=AsyncMock())
        self.bot.panel = NS(show=AsyncMock(), hidden=False)
        self.source = NS(text='', reply_text=AsyncMock(), edit_text=AsyncMock(), delete=AsyncMock())
        self.update = NS(effective_message=self.source, effective_user=NS(id=1),
                         effective_chat=NS(id=1, type='private'))
        self.context = NS(user_data={})

    def workspace(self, account, dialog):
        ws = Workspace(account, dialog, 'Test contact', self.root / f'account_{account}' / f'dialog_{dialog}')
        ws.root.mkdir(parents=True)
        return ws

    async def click(self, operation, key='', ws=None):
        action = button('test', operation, ws or self.ws, key).callback_data
        self.assertLessEqual(len(action.encode()), 64)
        await self.bot.callbacks.dispatch(action, self.update, self.context)

    async def text(self, value):
        self.source.text = value
        await self.bot.text_instruction(self.update, self.context)

    def test_store_restart_and_workspace_isolation(self):
        case = self.store.save_case(self.draft)
        self.assertEqual(BenchmarkStore(self.ws).case(case['id']), case)
        self.assertEqual(case['expected_source_message_ids'], [7])
        for account, dialog in ((2, 42), (1, 43)):
            other = self.workspace(account, dialog)
            self.assertEqual(BenchmarkStore(other).cases(), [])
            with self.assertRaises(ValueError):
                BenchmarkStore(other).fact(self.fact_id)
            # Even a mistakenly copied file must not be accepted in another workspace.
            (other.root / 'search_benchmark.json').write_bytes(self.store.path.read_bytes())
            with self.assertRaises(ValueError):
                BenchmarkStore(other).cases()

    def test_deleted_or_changed_source_blocks_save(self):
        mark_messages_deleted([7], filename=self.ws.deleted_message_ids)
        with self.assertRaises(ValueError):
            self.store.save_case(self.draft)
        self.assertEqual(self.store.cases(), [])

    def test_atomic_failure_preserves_old_file(self):
        original = self.store.path.read_bytes()
        with patch('mybot.storage.atomic.os.replace', side_effect=OSError('simulated crash')):
            with self.assertRaises(OSError):
                self.store.save_case(self.draft)
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertEqual(self.store.cases(), [])
        self.assertEqual(list(self.ws.root.glob('.search_benchmark.json.*')), [])

    def test_snapshot_rating_error_and_deletion(self):
        case = self.store.save_case(self.draft)
        run = self.store.save_run(case, 'test result', 'test-model')
        changed = self.store.save_case(dict(case, case_id=case['id'], question='Другой вопрос'))
        self.assertNotEqual(changed['question'], self.store.run(run['id'])['case_snapshot']['question'])
        self.store.rate(run['id'], 'correct')
        self.assertEqual(BenchmarkStore(self.ws).run(run['id'])['rating'], 'correct')
        self.store.rate(run['id'], 'wrong')
        self.assertEqual(self.store.latest()[case['id']]['rating'], 'wrong')
        error = self.store.save_run(changed, None, 'test-model', 'TimeoutError')
        with self.assertRaises(ValueError):
            self.store.rate(error['id'], 'wrong')
        self.assertEqual(self.store.latest()[case['id']]['status'], 'error')
        self.store.delete_case(case['id'])
        self.assertEqual(self.store.cases(), [])
        self.assertEqual(self.store.run(run['id'])['rating'], 'wrong')

    async def test_add_edit_confirm_cancel_and_stale_input(self):
        await self.click('add', self.fact_id)
        await self.text('Какой любимый цвет?')
        self.assertEqual(self.store.cases(), [])
        draft = self.context.user_data['benchmark_draft']
        await self.click('answer', draft['token'])
        await self.text('Зелёный.')
        await self.click('save', draft['token'])
        case = self.store.cases()[0]
        self.assertEqual(case['expected_answer'], 'Зелёный.')
        self.assertNotIn('benchmark_draft', self.context.user_data)
        await self.click('edit', case['id'])
        token = self.context.user_data['benchmark_draft']['token']
        await self.click('question', token)
        await self.text('Новая формулировка?')
        await self.click('cancel')
        self.assertEqual(self.store.case(case['id'])['question'], 'Какой любимый цвет?')
        await self.click('save', token)
        self.assertIn('устарел', self.bot.menu.show.await_args.args[1])

    async def test_home_start_and_dialog_change_cancel_input(self):
        for action in ('home', 'start', 'switch'):
            self.bot.workspace = self.ws
            await self.click('add', self.fact_id)
            if action == 'home':
                await self.bot.callbacks.dispatch('ui:home', self.update, self.context)
            elif action == 'start':
                await self.bot.start_command(self.update, self.context)
            else:
                self.bot.workspace = self.workspace(1, 99)
                await self.text('Should not be saved')
            self.assertNotIn('benchmark_input', self.context.user_data)
            self.assertEqual(self.store.cases(), [])

    async def test_evaluation_passes_only_question_requires_confirmation_and_is_read_only(self):
        case = self.store.save_case(self.draft)
        history = self.ws.chat_history.read_bytes()
        with patch('mybot.telegram.bot_benchmark.search_history', new=AsyncMock(return_value='Green. ID 7')) as search:
            await self.click('run', case['id'])
            search.assert_not_awaited()
            await self.click('check', case['id'])
            self.assertIn('платные', self.bot.menu.show.await_args.args[1])
            await self.click('run', case['id'])
            search.assert_awaited_once_with(self.ws, case['question'], persist_cache=False)
            await self.click('run', case['id'])
            search.assert_awaited_once()
        run = self.store.latest()[case['id']]
        self.assertEqual(run['case_snapshot'], case)
        self.assertIsNone(run['rating'])
        await self.click('rate_ok', run['id'])
        self.assertEqual(self.store.run(run['id'])['rating'], 'correct')
        self.assertEqual(self.ws.chat_history.read_bytes(), history)
        self.assertEqual({p.name for p in self.ws.root.iterdir()},
                         {'chat_history.jsonl', 'search_benchmark.json', 'search_benchmark_runs.json'})
        self.assertIn('Green. ID 7', self.source.reply_text.await_args.args[0])

    async def test_api_error_has_no_quality_rating_and_can_be_reopened(self):
        case = self.store.save_case(self.draft)
        with patch('mybot.telegram.bot_benchmark.search_history', new=AsyncMock(side_effect=TimeoutError())), \
             self.assertLogs(level='ERROR'):
            await self.click('check', case['id'])
            await self.click('run', case['id'])
        run = BenchmarkStore(self.ws).latest()[case['id']]
        self.assertEqual(run['status'], 'error')
        self.assertIsNone(run['rating'])
        await self.click('result', run['id'])
        self.assertIn('TimeoutError', self.bot.menu.show.await_args.args[1])
        await self.click('list')
        self.assertIn('Ошибка выполнения: 1', self.bot.menu.show.await_args.args[1])

    async def test_delete_needs_confirmation_and_old_buttons_do_not_access_other_dialog(self):
        case = self.store.save_case(self.draft)
        await self.click('delete_yes', case['id'])
        self.assertEqual(len(self.store.cases()), 1)
        await self.click('delete', case['id'])
        await self.click('delete_yes', case['id'])
        self.assertEqual(self.store.cases(), [])
        await self.click('view', case['id'])
        self.assertIn('недоступна', self.bot.menu.show.await_args.args[1])
        self.bot.workspace = self.workspace(1, 55)
        await self.click('add', self.fact_id)
        self.assertNotIn('benchmark_draft', self.context.user_data)
        self.assertIn('другому диалогу', self.bot.menu.show.await_args.args[1])

    async def test_read_only_search_does_not_create_tail_cache(self):
        before = {p.name: p.read_bytes() for p in self.ws.root.iterdir()}
        model_answer = NS(output_text=json.dumps({'candidates': [{'answer': 'Green', 'source_ids': [7]}]}))
        with patch('mybot.services.history_search.ai_client.embeddings.create',
                   new=AsyncMock(return_value=NS(data=[NS(embedding=[1.0, 0.0])]))), \
             patch('mybot.services.history_search.create_query_embedding', new=AsyncMock(return_value=[1.0, 0.0])), \
             patch('mybot.services.history_search.ai_client.responses.create', new=AsyncMock(return_value=model_answer)):
            result = await search_history(self.ws, 'colour?', persist_cache=False)
        self.assertIn('Green', result)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.ws.root.iterdir()})

    def test_private_paths_are_git_ignored(self):
        result = subprocess.run(['git', 'check-ignore', '--stdin'], input=(
            'data/account_1/dialogs/dialog_42/search_benchmark.json\n'
            'data/account_1/dialogs/dialog_42/search_benchmark_runs.json\n'),
            text=True, capture_output=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(result.stdout.splitlines()), 2)

    def test_fact_and_evaluation_cards_are_not_transient_menus(self):
        for operation in ('add', 'rate_ok', 'rate_bad', 'rate_none'):
            keyboard = NS(inline_keyboard=[[button('test', operation, self.ws, 'a' * 16)]])
            self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=keyboard)))
            raw = keyboard.inline_keyboard[0][0].callback_data.encode()
            self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=NS(rows=[NS(buttons=[NS(data=raw)])]))))


if __name__ == '__main__':
    unittest.main()

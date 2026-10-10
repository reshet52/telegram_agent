"""Review changes, deleted sources and overlapping contexts must fail closed."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.services.reviewed_dataset import prepare_reviewed, load_reviewed, reviewed_examples
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_learning import handle_learning


class ReviewedTrainingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.ws = Workspace(1, 10, 'Test', Path(temporary.name))
        messages = [dict(message_id=i, sender='Я' if i % 2 == 0 else 'Other',
                         text=f'ordinary message {i}', type='text') for i in range(1, 601)]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
        self.store = PracticeStore(self.ws)
        self.keys = []
        for target in range(30, 391, 40):
            with patch('mybot.storage.reply_practice.random.choice',
                       side_effect=lambda items, t=target: next(x for x in items if x[1][0]['message_id'] == t)):
                case = self.store.create()
            self.keys.append(case['id'])
            self.store.save_result(case['id'], [f'answer {target}', 'second', 'third'])
            self.store.rate(case['id'], 1, 'good')

    async def test_source_disjoint_targets_and_revocation(self):
        manifest = prepare_reviewed(self.ws)
        _, rows = load_reviewed(self.ws, manifest['id'])
        self.assertEqual({k: len(v) for k, v in rows.items()}, {'train': 8, 'valid': 2})
        self.assertEqual(rows['train'][0]['completion'][0]['content'], 'answer 30')
        train_ids = {i for item in manifest['examples']['train'] for i in item['ids']}
        valid_ids = {i for item in manifest['examples']['valid'] for i in item['ids']}
        self.assertFalse(train_ids & valid_ids)
        self.store.clear_learning(self.keys[0])
        with self.assertRaisesRegex(ValueError, 'изменились'):
            load_reviewed(self.ws, manifest['id'])

    async def test_correction_precedes_original_and_selected_variant(self):
        key = self.keys[0]
        self.store.reveal(key)
        self.store.use_original(key, True)
        self.assertEqual(reviewed_examples(self.ws)[0]['source_kind'], 'historical_sent_reply')
        self.store.save_correction(key, 'owner revision')
        item = reviewed_examples(self.ws)[0]
        self.assertEqual(item['source_kind'], 'owner_correction_not_sent')
        self.assertEqual(item['row']['completion'][0]['content'], 'owner revision')

    async def test_changed_source_and_wrong_account_rejected(self):
        manifest = prepare_reviewed(self.ws)
        with self.assertRaises(ValueError):
            load_reviewed(Workspace(2, 10, 'Other', self.ws.root), manifest['id'])
        mark_messages_deleted({30}, filename=self.ws.deleted_message_ids)
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, manifest['id'])

    async def test_too_few_ratings_do_not_start_training(self):
        self.store.clear_learning(self.keys[0])
        bot = NS(is_owner=lambda u: True, goal=NS(workspace=lambda: self.ws),
                 runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()))
        with patch('mybot.telegram.bot_learning.available', return_value=True), \
             patch('mybot.telegram.bot_learning.run_local_training', new_callable=AsyncMock) as run:
            await handle_learning(bot, 'ui:lt:1:10', None)
            run.assert_not_awaited()
        self.assertIn('Выбрано 9', bot.callbacks.show.await_args.args[1])

    async def test_wrong_dialog_and_nonowner_cannot_train(self):
        bot = NS(is_owner=lambda u: True, goal=NS(workspace=lambda: self.ws),
                 runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()))
        with patch('mybot.telegram.bot_learning.run_local_training', new_callable=AsyncMock) as run:
            await handle_learning(bot, 'ui:lt:2:10', None)
            bot.is_owner = lambda u: False
            await handle_learning(bot, 'ui:lt:1:10', None)
            run.assert_not_awaited()

    async def test_completed_training_reports_measured_losses(self):
        bot = NS(is_owner=lambda u: True, goal=NS(workspace=lambda: self.ws),
                 runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()))
        with patch('mybot.telegram.bot_learning.available', return_value=True), \
             patch('mybot.telegram.bot_learning.run_local_training', new_callable=AsyncMock,
                   return_value={'steps': 50, 'loss_before': 2.0, 'loss_after': 1.9}) as run:
            await handle_learning(bot, 'ui:lt:1:10', None)
            run.assert_awaited_once()
        self.assertIn('2.000 → 1.900', bot.callbacks.show.await_args.args[1])

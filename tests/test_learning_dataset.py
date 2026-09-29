"""Offline tests of chronological splits, reserved exercises and training wiring."""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.services.learning_dataset import prepare_dataset, load_dataset
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_learning import handle_learning
from scripts.train_reply_adapter import train


class DatasetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = Workspace(1, 10, 'Test', self.root / 'dialog')
        self.ws.root.mkdir()
        self.messages = [dict(message_id=i, sender='Я' if i % 2 == 0 else 'Other',
                              text=f'text {i}', type='text', date='2025-01-01') for i in range(1, 241)]
        self.write_history()

    def write_history(self):
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))

    async def test_source_disjoint_split_and_target_is_real_outgoing(self):
        manifest = prepare_dataset(self.ws)
        checked, rows = load_dataset(self.ws, manifest['id'])
        self.assertEqual(manifest, checked)
        self.assertTrue(rows['train'] and rows['valid'])
        root = self.ws.root / 'learning/datasets' / manifest['id']
        sources = json.loads((root / 'sources.json').read_text())
        train_ids = {i for row in sources['train'] for i in row['context_ids'] + row['response_ids']}
        valid_ids = {i for row in sources['valid'] for i in row['context_ids'] + row['response_ids']}
        self.assertFalse(train_ids & valid_ids)
        self.assertLess(max(train_ids), min(valid_ids))
        for row, source in zip(rows['train'], sources['train']):
            self.assertEqual(row['completion'], [{'role': 'assistant',
                                                  'content': f"text {source['response_ids'][0]}"}])
            self.assertEqual(row['prompt'][-1]['role'], 'user')
        original = self.ws.chat_history.read_text()
        second = prepare_dataset(self.ws)
        self.assertNotEqual(second['id'], manifest['id'])
        self.assertEqual(self.ws.chat_history.read_text(), original)

    async def test_practice_sources_are_reserved_and_new_practice_invalidates_old_dataset(self):
        manifest = prepare_dataset(self.ws)
        with patch('mybot.storage.reply_practice.random.choice', side_effect=lambda items: items[0]):
            case = PracticeStore(self.ws).create()
        with self.assertRaisesRegex(ValueError, 'пересекается'):
            load_dataset(self.ws, manifest['id'])
        new = prepare_dataset(self.ws)
        self.assertIn(case['expected_ids'][0], new['reserved_practice_ids'])
        load_dataset(self.ws, new['id'])

    async def test_modified_or_deleted_source_and_wrong_account_rejected(self):
        manifest = prepare_dataset(self.ws)
        wrong = Workspace(2, 10, 'Other', self.ws.root)
        with self.assertRaises(ValueError):
            load_dataset(wrong, manifest['id'])
        self.messages[0]['text'] = 'changed'
        self.write_history()
        with self.assertRaises(ValueError):
            load_dataset(self.ws, manifest['id'])
        self.messages[0]['text'] = 'text 1'
        self.write_history()
        mark_messages_deleted({2}, filename=self.ws.deleted_message_ids)
        with self.assertRaises(ValueError):
            load_dataset(self.ws, manifest['id'])

    async def test_partial_export_and_tampering_fail_closed(self):
        manifest = prepare_dataset(self.ws)
        root = self.ws.root / 'learning/datasets' / manifest['id']
        (root / 'train.jsonl').write_text('{}\n')
        with self.assertRaises(ValueError):
            load_dataset(self.ws, manifest['id'])
        with self.assertRaises(ValueError):
            load_dataset(self.ws, '../escape')
        with patch('mybot.services.learning_dataset.write_json', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                prepare_dataset(self.ws)
        partial = next(path.name for path in root.parent.iterdir() if path.name != manifest['id'])
        with self.assertRaises(ValueError):
            load_dataset(self.ws, partial)

    async def test_short_history_is_not_silently_used_as_training_and_test(self):
        self.messages = self.messages[:8]
        self.write_history()
        with self.assertRaises(ValueError):
            prepare_dataset(self.ws)
        self.assertFalse((self.ws.root / 'learning').exists())

    async def test_telegram_preparation_owner_and_stale_button_guards(self):
        bot = NS(is_owner=lambda update: True, goal=NS(workspace=lambda: self.ws),
                 runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()))
        await handle_learning(bot, 'ui:learning', None)
        self.assertFalse((self.ws.root / 'learning').exists())
        await handle_learning(bot, 'ui:ld:2:10', None)
        self.assertFalse((self.ws.root / 'learning').exists())
        bot.is_owner = lambda update: False
        await handle_learning(bot, 'ui:ld:1:10', None)
        self.assertFalse((self.ws.root / 'learning').exists())
        bot.is_owner = lambda update: True
        await handle_learning(bot, 'ui:ld:1:10', None)
        self.assertIn('Набор готов', bot.callbacks.show.await_args.args[1])

    async def test_lora_runner_completion_only_offline_and_no_automatic_activation(self):
        manifest = prepare_dataset(self.ws)
        model_path = self.root / 'base'
        model_path.mkdir()
        tokenizer = Mock(chat_template='template', pad_token_id=0)
        tokenizer.apply_chat_template.return_value = [1, 2, 3]
        model = NS(config=NS(use_cache=True))
        trainer = Mock()
        trainer.evaluate.side_effect = [{'eval_loss': 2.0}, {'eval_loss': 1.8}]
        modules = {
            'datasets': NS(Dataset=NS(from_list=lambda rows: rows)),
            'peft': NS(LoraConfig=Mock()),
            'transformers': NS(AutoTokenizer=NS(from_pretrained=Mock(return_value=tokenizer)),
                               AutoModelForCausalLM=NS(from_pretrained=Mock(return_value=model))),
            'trl': NS(SFTConfig=Mock(), SFTTrainer=Mock(return_value=trainer)),
        }
        with patch.dict(sys.modules, modules):
            root = train(self.ws, manifest['id'], model_path, steps=1)
        config = modules['trl'].SFTConfig.call_args.kwargs
        self.assertTrue(config['completion_only_loss'])
        self.assertFalse(config['push_to_hub'])
        self.assertEqual(config['report_to'], 'none')
        self.assertTrue(modules['transformers'].AutoModelForCausalLM.from_pretrained.call_args.kwargs['local_files_only'])
        trainer.train.assert_called_once()
        self.assertEqual(trainer.evaluate.call_count, 2)
        run = json.loads((root / 'run.json').read_text())
        self.assertEqual(run['status'], 'trained_unapproved')
        self.assertEqual(run['dialog_id'], self.ws.dialog_id)
        self.assertEqual(run['before']['eval_loss'], 2.0)

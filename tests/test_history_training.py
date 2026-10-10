import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mybot.services.history_sampling import balanced
from mybot.services.history_training import prepare
from mybot.services.reviewed_dataset import load_reviewed
from mybot.services import style_versions, training_jobs
from mybot.services.reply_practice import generate_practice
from mybot.storage.atomic import write_json
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.workspace import Workspace


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages) + '<|im_start|>assistant\n'

    def encode(self, text, **kwargs):
        result = list(text.encode())
        return result + [0] * 2048 if text.endswith('LONG<|im_end|>') else result


class HistoryTrainingTests(unittest.IsolatedAsyncioTestCase):
    async def test_action_gate_blocks_new_adapter_with_invalid_decisions(self):
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        path = style_versions.quality_path(self.ws, version)
        from mybot.storage.atomic import read_json
        report = read_json(path)
        report['valid_actions'] = 9
        write_json(path, report)
        self.assertFalse(style_versions.quality_passed(self.ws, version))
        self.assertEqual(style_versions.practice_selected(self.ws), 'base')

    async def test_confirmed_wait_replaces_conflicting_historical_replay(self):
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        self.adapter(manifest)
        source = next(item for item in manifest['examples']['train']
                      if item['source_kind'] == 'historical_sent_reply')
        history = {m['message_id']: m for m in self.lesson.store.history()}
        case = self.lesson.store.create_from([history[i] for i in source['context_ids']],
                                             [history[i] for i in source['response_ids']])
        self.lesson.store.confirm_action(case['id'], 'wait')
        refined = prepare(self.ws, Tokenizer(), self.ws.root, purpose='refine', limit=30)
        train = refined['examples']['train']
        self.assertTrue(any(item['case_id'] == 'action:' + case['id']
                            and item['row']['completion'][0]['content'] == 'WAIT' for item in train))
        self.assertFalse(any(item['case_id'].startswith(('history:', 'history_action:'))
                             and set(item['response_ids']) & set(source['response_ids']) for item in train))
        load_reviewed(self.ws, refined['id'])

    async def test_quality_gate_blocks_activation_without_changing_live_pointer(self):
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        quality = style_versions.quality_path(self.ws, version)
        from mybot.storage.atomic import read_json
        report = read_json(quality)
        report['status'] = 'failed'
        write_json(quality, report)
        self.assertEqual(style_versions.practice_selected(self.ws), 'base')
        self.assertIn('не прошёл', style_versions.practice_label(self.ws))
        self.assertEqual(style_versions.selected(self.ws), 'configured')
        with self.assertRaises(ValueError):
            style_versions.activate_practice(self.ws, version)
        with self.assertRaises(ValueError):
            style_versions.activate(self.ws, version)
        self.assertTrue(style_versions.adapter(self.ws, version, require_quality=False))

    async def test_repair_keeps_sources_and_old_comparison_unchanged(self):
        from mybot.services.history_training import prepare_repair
        from mybot.services.style_comparison import open_comparison
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        old = self.lesson.root / 'comparisons' / (version + '.json')
        write_json(old, {'legacy': True, 'votes': {'0': 'neither'}})
        before = old.read_bytes()
        new = open_comparison(self.ws, version)
        self.assertEqual(new['votes'], {})
        self.assertEqual(old.read_bytes(), before)
        repaired = prepare_repair(self.ws, Tokenizer(), self.ws.root, version)
        self.assertNotEqual(repaired['id'], manifest['id'])
        self.assertEqual(repaired['examples'], manifest['examples'])
        self.assertIsNone(repaired['parent_adapter'])
        self.assertEqual(repaired['learning_rate'], 2e-5)
        self.assertEqual(repaired['repair_of'], version)
        load_reviewed(self.ws, manifest['id'])
        load_reviewed(self.ws, repaired['id'])

    async def test_legacy_comparison_button_cannot_vote_in_new_comparison(self):
        import asyncio
        from types import SimpleNamespace as NS
        from mybot.telegram.bot_style_lesson import BotStyleLesson, button
        from mybot.services.style_comparison import open_comparison
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        open_comparison(self.ws, version)
        api = NS(send_message=AsyncMock(return_value=NS(message_id=77)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda _: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        await ui.handle(f'ui:sl:vote:{version}:0:first', None, None)
        self.assertIn('прежнее сравнение', api.send_message.call_args.kwargs['text'])
        self.assertEqual(open_comparison(self.ws, version)['votes'], {})
        self.assertTrue(button('Vote', 'vote', f'{version}:0:first').callback_data.startswith('ui:s2:'))

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ws = Workspace(7, 9, 'Test', Path(temp.name))
        self.messages = [dict(message_id=i, sender='Я' if i % 2 == 0 else 'Other',
                              text=f'message-{i}', date=f'2025-{1 + i // 150:02d}-01', type='text')
                         for i in range(1, 1001)]
        self.save_history()
        self.lesson = StyleLesson(self.ws)

    def save_history(self):
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))

    def adapter(self, manifest):
        version = 'a' * 32
        root = self.ws.root / 'learning/mlx_runs' / version
        root.mkdir(parents=True)
        (root / 'adapters.safetensors').write_bytes(b'adapter')
        write_json(root / 'run.json', dict(account_id=7, dialog_id=9, status='trained_unapproved',
            purpose='history', dataset_id=manifest['id'], adapter_sha256=hashlib.sha256(b'adapter').hexdigest()))
        from mybot.ai.local_format import FORMAT_VERSION
        write_json(style_versions.quality_path(self.ws, version), dict(account_id=7, dialog_id=9,
            status='passed', format_version=FORMAT_VERSION, adapter_sha256=hashlib.sha256(b'adapter').hexdigest(),
            cases=self.lesson.holdout()['cases'], counts={'candidate': 10}, expected_per_model=10,
            valid_actions=10, expected_actions=10))
        style_versions.activate_practice(self.ws, version)
        return version

    def test_balance_and_redistribution(self):
        items = [{'period': i, 'n': n} for i, count in enumerate([1, 20, 20]) for n in range(count)]
        chosen = balanced(items, 12)
        self.assertEqual([sum(x['period'] == i for x in chosen) for i in range(3)], [1, 6, 5])
        self.assertEqual(len({(x['period'], x['n']) for x in chosen}), 12)

    def test_bootstrap_no_reviews_holdout_and_revalidate(self):
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        self.assertEqual(manifest['period_counts'], [10, 10, 10])
        self.assertEqual(manifest['reviewed_count'], 0)
        holdout = self.lesson.holdout()
        _, rows = load_reviewed(self.ws, manifest['id'])
        for item in manifest['examples']['train']:
            self.assertIn(item['source_kind'], {'historical_sent_reply', 'historical_observed_reply_action'})
            self.assertLess(max(item['ids']), min(self.lesson.reserved_ids()))
            prompt = json.dumps(item['row']['prompt'])
            for source_id in item['response_ids']:
                self.assertNotIn(f'message-{source_id}"', prompt)
        self.assertEqual(len(rows['train']), 30)
        self.assertEqual(holdout, self.lesson.holdout())
        with self.assertRaises(ValueError):
            load_reviewed(Workspace(8, 9, 'wrong', self.ws.root), manifest['id'])
        target = manifest['examples']['train'][0]['response_ids'][0]
        self.messages[target - 1]['text'] = 'edited'
        self.save_history()
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, manifest['id'])

    def test_length_filter_precedes_sampling_and_shortage(self):
        # Remove most of the early period through token eligibility, then redistribute.
        for m in self.messages[:280]:
            if m['sender'] == 'Я':
                m['text'] += 'LONG'
        self.save_history()
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=90)
        self.assertEqual(manifest['counts']['train'], 90)
        self.assertLess(manifest['period_counts'][0], 30)
        self.assertTrue(all(not x['row']['completion'][0]['content'].endswith('LONG')
                            for x in manifest['examples']['train']))
        for m in self.messages:
            m['text'] += 'LONG'
        self.save_history()
        # Existing holdout source edits must fail rather than silently replace the holdout.
        with self.assertRaises(ValueError):
            prepare(self.ws, Tokenizer(), self.ws.root)

    async def test_practice_pointer_cache_and_refine_replay(self):
        state = self.lesson.start()
        key = state['cases'][0]
        with self.assertRaisesRegex(ValueError, 'обученная версия'):
            await generate_practice(self.lesson.store, key, local=True)
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['a', 'b'])):
            old = await generate_practice(self.lesson.store, key, local=True)
        self.assertEqual(style_versions.selected(self.ws), 'configured')
        self.assertEqual(style_versions.practice_selected(self.ws), version)
        self.assertEqual(self.lesson.current(), state)
        self.assertEqual(await generate_practice(self.lesson.store, key, local=True), old)
        next_key = state['cases'][1]
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['a', 'b'])) as gen:
            await generate_practice(self.lesson.store, next_key, local=True)
            self.assertEqual(gen.call_args.args[0]['version'], version)
        self.lesson.store.save_correction(key, 'My correction')
        refined = prepare(self.ws, Tokenizer(), self.ws.root, purpose='refine', limit=30)
        self.assertEqual(refined['parent_adapter'], version)
        self.assertEqual(refined['reviewed_count'], 1)
        original_ids = {x['case_id'] for x in manifest['examples']['train']}
        replay = [x for x in refined['examples']['train'] if x['case_id'].startswith('history:')]
        self.assertTrue(all(x['case_id'] in original_ids for x in replay))
        load_reviewed(self.ws, refined['id'])
        self.lesson.store.clear_learning(key)
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, refined['id'])

    def test_lesson_rotates_extra_and_resumes(self):
        for expected in ([3, 3, 4], [4, 3, 3], [3, 4, 3]):
            state = self.lesson.start()
            periods = [self.lesson.store.get(k)['period'] for k in state['cases']]
            self.assertEqual([periods.count(i) for i in range(3)], expected)
            self.lesson.pause()
            self.assertEqual(self.lesson.start()['cases'], state['cases'])
            for key in state['cases']:
                self.lesson.advance(key, 'skip')

    async def test_preview_no_training_stale_confirmation(self):
        from types import SimpleNamespace
        process = SimpleNamespace(pid=1234, returncode=None, wait=AsyncMock(return_value=0))
        with patch('mybot.services.training_jobs.available', return_value=True), \
             patch('mybot.services.training_jobs.asyncio.create_subprocess_exec', AsyncMock(return_value=process)) as spawn:
            job = await training_jobs.preview_history(self.ws)
            again = await training_jobs.preview_history(self.ws)
            self.assertEqual(job['id'], again['id'])
            self.assertIn('scripts.prepare_style_history', spawn.call_args.args)
            self.assertNotIn('--dataset', spawn.call_args.args)
            spawn.assert_awaited_once()
            process.returncode = 0
            with self.assertRaises(ValueError):
                await training_jobs.start(self.ws, 'b' * 32)
        training_jobs._processes.pop(job['id'], None)

    async def test_preview_restart_and_ui_confirmation(self):
        from types import SimpleNamespace as NS
        from mybot.telegram.bot_style_lesson import BotStyleLesson
        import asyncio
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        key = 'b' * 32
        summary = {k: manifest[k] for k in ('purpose', 'counts', 'period_counts', 'reviewed_count', 'dates', 'estimated_minutes')}
        self.lesson.write('training_job.json', dict(id=key, kind='preview', purpose='history', pid=None, state='preparing'))
        write_json(self.lesson.root / 'mlx_runs' / key / 'run.json', dict(status='prepared', dataset_id=manifest['id'], summary=summary))
        self.assertEqual(training_jobs.status(self.ws)['state'], 'preview_ready')
        api = NS(send_message=AsyncMock(return_value=NS(message_id=55)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda u: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        await ui.handle('ui:sl:job:7:9', None, None)
        content = api.send_message.call_args.kwargs
        self.assertIn('Обучение: 30', content['text'])
        self.assertIn('10/10/10', content['text'])
        callbacks = [b.callback_data for row in content['reply_markup'].inline_keyboard for b in row]
        self.assertIn('ui:sl:confirm:' + manifest['id'], callbacks)
        with patch('mybot.services.training_jobs.available', return_value=True), \
             patch('mybot.services.training_jobs.spawn', AsyncMock(return_value={})) as spawn:
            await training_jobs.start(self.ws, manifest['id'])
            self.assertEqual(spawn.call_args.args[1]['baseline'], 'base')
            self.assertEqual(spawn.call_args.args[2], 'scripts.train_reply_mlx')

    async def test_preview_stop_and_identity_after_restart(self):
        from types import SimpleNamespace as NS
        key = 'c' * 32
        job = dict(id=key, kind='preview', state='preparing', pid=99999)
        self.lesson.write('training_job.json', job)
        command = 'python -m scripts.prepare_style_history --run-id ' + key
        with patch('mybot.services.training_jobs.subprocess.run', return_value=NS(stdout=command)), \
             patch('mybot.services.training_jobs.os.kill') as kill, \
             patch('mybot.services.training_jobs.force_stop', AsyncMock()):
            self.assertTrue(training_jobs.alive(job))
            self.assertEqual(training_jobs.stop(self.ws)['state'], 'stopping')
            kill.assert_called_once()
        with patch('mybot.services.training_jobs.subprocess.run', return_value=NS(stdout='')):
            self.assertEqual(training_jobs.status(self.ws)['state'], 'stopped')

    async def test_blind_issues_stay_out_of_training(self):
        from mybot.services import style_comparison
        manifest = prepare(self.ws, Tokenizer(), self.ws.root, limit=30)
        version = self.adapter(manifest)
        with patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['reply'])):
            data = await style_comparison.generate_pair(self.ws, version, 0)
        side = data['orders']['0'][0]
        data = style_comparison.flag(self.ws, version, 0, 0, 'emoji')
        self.assertEqual(data['issues']['0'][side], ['emoji'])
        self.assertIn('Чужие эмодзи:', style_comparison.issue_summary(data))
        data = style_comparison.flag(self.ws, version, 0, 0, 'emoji')
        self.assertEqual(data['issues']['0'][side], [])
        from mybot.services.reviewed_dataset import training_examples
        self.assertEqual(training_examples(self.ws), [])

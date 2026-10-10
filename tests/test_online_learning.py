import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock

from mybot.services.online_learning import collect, enable, read_state, state_path
from mybot.storage.atomic import write_json
from mybot.storage.reply_journal import (record_generation, record_outgoing_reply, create_draft,
                                       mark_draft_kept, link_outgoing_to_draft)
from mybot.storage.workspace import Workspace


class ManualStartTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_start_bypasses_idle_but_not_active_generation(self):
        import time
        from mybot.services import online_learning as learning
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            state = {'enabled': True, 'enabled_at': time.time(), 'queue': [{}] * 10, 'consumed': [], 'excluded': []}
            write_json(ws.root / 'learning/online_request.json', {'at': time.time(), 'active': 0})
            with patch.object(learning, 'collect', return_value=state), \
                 patch('mybot.services.style_versions.selected', return_value='a' * 32), \
                 patch.object(learning, '_processes', {}), \
                 patch('mybot.services.online_learning.asyncio.create_subprocess_exec', AsyncMock()) as spawn:
                await learning.tick(ws)
                spawn.assert_not_awaited()
                write_json(ws.root / 'learning/online_request.json', {'at': time.time(), 'active': 1})
                self.assertIn('готовятся ответы', await learning.tick(ws, manual=True))
                spawn.assert_not_awaited()
                write_json(ws.root / 'learning/online_request.json', {'at': time.time(), 'active': 0})
                self.assertEqual(await learning.tick(ws, manual=True), 'Обучение запущено.')
                spawn.assert_awaited_once()


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'test', Path(self.temp.name) / 'one')
        self.ws.root.mkdir()
        self.start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.messages = []
        self.append(1, 'Other', 'question', 0)
        enable(self.ws, clock=lambda: self.start.timestamp())

    def append(self, number, sender, text, seconds):
        self.messages.append(dict(message_id=number, sender=sender, text=text, type='text',
                                  date=(self.start + timedelta(seconds=seconds)).isoformat()))
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))

    def scan(self, seconds):
        with patch('mybot.services.online_learning.reserved_ids', return_value=set()):
            return collect(self.ws, clock=lambda: self.start.timestamp() + seconds)

    def test_silence_grouping_and_restart_dedup(self):
        self.append(2, 'Я', 'part1', 1)
        self.append(3, 'Я', 'part2', 10)
        self.assertEqual(len(self.scan(39)['queue']), 0)
        state = self.scan(40)
        self.assertEqual(state['queue'][0]['response_ids'], [2, 3])
        self.assertEqual(state['queue'][0]['text'], 'part1\n\npart2')
        self.assertEqual(len(self.scan(100)['queue']), 1)
        state['consumed'] = [state['queue'][0]['id']]
        state['queue'] = []
        write_json(state_path(self.ws), state)
        self.assertEqual(len(self.scan(200)['queue']), 0)

    def test_incoming_closes_group_and_independent_reply_qualifies(self):
        self.append(2, 'Я', 'my independent answer', 1)
        self.append(3, 'Other', 'next question', 2)
        state = self.scan(3)
        self.assertEqual(state['queue'][0]['response_ids'], [2])
        self.assertIsNone(state['queue'][0]['provenance'])

    def test_older_media_keeps_valid_text_suffix_and_recovers_old_exclusion(self):
        self.messages[0]['type'] = 'photo'
        self.append(2, 'Other', 'latest question', 1)
        self.append(3, 'Я', 'sent answer', 2)
        from mybot.storage.reply_practice import digest
        state = read_state(self.ws)
        state['excluded'] = [{'id': digest([3]), 'response_ids': [3], 'reason': 'nontext_source'}]
        write_json(state_path(self.ws), state)
        state = self.scan(40)
        self.assertEqual(len(state['queue']), 1)
        self.assertEqual(state['queue'][0]['context_ids'], [2])
        self.assertEqual(state['excluded'], [])
        self.assertEqual(len(self.scan(100)['queue']), 1)

    def test_actual_sent_api_answer_keeps_explicit_link(self):
        key = record_generation(self.ws, ['a', 'b', 'c', 'api answer'], 1, 'mixed')
        draft = create_draft(self.ws, key, 4)
        mark_draft_kept(self.ws, draft)
        self.assertEqual(self.scan(100)['queue'], [])
        self.append(2, 'Я', 'api answer', 1)
        record_outgoing_reply(self.ws, self.messages[-1])
        link_outgoing_to_draft(self.ws, 2, draft)
        state = self.scan(40)
        self.assertEqual(state['queue'][0]['provenance']['chosen_variant'], 4)

    def test_deleted_reserved_and_cross_dialog_sources_excluded(self):
        self.append(2, 'Я', 'answer', 1)
        with patch('mybot.services.online_learning.reserved_ids', return_value={1}):
            state = collect(self.ws, clock=lambda: self.start.timestamp() + 40)
        self.assertEqual(state['queue'], [])
        self.assertEqual(state['excluded'][0]['reason'], 'deleted_or_reserved_source')
        other = Workspace(1, 3, 'other', self.ws.root.parent / 'other')
        other.root.mkdir()
        other.chat_history.write_text('')
        enable(other)
        self.assertEqual(read_state(other)['queue'], [])
        write_json(state_path(other), read_state(self.ws))
        with self.assertRaises(ValueError):
            read_state(other)


class IncrementalDatasetTests(unittest.TestCase):
    def setUp(self):
        from tests.test_full_history_training import Tokenizer
        self.tokenizer = Tokenizer()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'synthetic', Path(self.temp.name))
        self.messages = [dict(message_id=i, sender='Other' if i % 2 else 'Я',
                              text='question' if i % 2 else 'answer', type='text',
                              date='2025-01-01T00:00:00+00:00') for i in range(1, 25)]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))
        from mybot.services.learning_dataset import example
        from mybot.storage.reply_practice import digest
        def item(a, b):
            return {'case_id': f'full:{b}', 'context_ids': [a], 'response_ids': [b], 'ids': [a, b],
                    'source_digest': digest(self.messages[a-1:b]),
                    'row': example([self.messages[a-1]], [self.messages[b-1]])}
        self.original = {'max_length': 2048, 'holdout_ref': 'holdout.json',
                         'examples': {'train': [item(3, 4)], 'valid': [item(1, 2)]}}
        self.pending = [{'id': str(b), 'context_ids': [b-1], 'response_ids': [b],
                         'source_digest': digest(self.messages[b-2:b])} for b in range(6, 25, 2)]
        self.parent = self.ws.root / 'parent'
        self.parent.mkdir()
        write_json(self.parent / 'run.json', {'dataset_id': 'a'*32, 'adapter_sha256': 'parent-hash'})

    def prepare(self):
        from mybot.services.online_dataset import prepare
        with patch('mybot.services.style_versions.adapter', return_value=self.parent), patch(
                'mybot.services.reviewed_dataset.load_reviewed', return_value=(self.original, {})), patch(
                'mybot.services.online_dataset.reserved_ids', return_value={1, 2}), patch(
                'mybot.services.online_dataset.tokenizer_hash', return_value='tokenizer'):
            return prepare(self.ws, self.pending, 'b'*32, self.tokenizer, Path('model'))

    def test_single_epoch_replay_and_frozen_validation(self):
        manifest, excluded = self.prepare()
        self.assertEqual(excluded, [])
        self.assertEqual(len(manifest['new_events']), 10)
        self.assertEqual(manifest['counts']['train'], 11)
        self.assertEqual(sorted(manifest['order']), list(range(11)))
        self.assertEqual(manifest['examples']['valid'], self.original['examples']['valid'])

    def test_resume_refuses_changed_actual_reply(self):
        from mybot.services.online_dataset import load
        manifest, _ = self.prepare()
        self.messages[-1]['text'] = 'edited source'
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))
        with patch('mybot.services.online_dataset.reserved_ids', return_value={1, 2}), self.assertRaises(ValueError):
            load(self.ws, manifest)

    def test_oversized_target_explicitly_excluded_without_truncation(self):
        from mybot.storage.reply_practice import digest
        self.messages[-1]['text'] = 'x'*3000
        self.pending[-1]['source_digest'] = digest(self.messages[-2:])
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))
        manifest, excluded = self.prepare()
        self.assertIsNone(manifest)
        self.assertEqual(excluded[-1]['reason'], 'full_target_does_not_fit')

    def test_confirmed_api_target_replaces_conflicting_historical_replay(self):
        from mybot.services.simple_lessons import confirm
        from mybot.storage.reply_practice import PracticeStore
        from mybot.storage.style_lesson import StyleLesson
        from mybot.services.online_learning import read_state
        case = PracticeStore(self.ws).create_from([self.messages[2]], [self.messages[3]], 'simple_lesson')
        StyleLesson(self.ws).write('simple_lesson.json', {'cases': [case['id']], 'index': 0, 'choices': {}})
        with patch('mybot.services.simple_lessons.reserved_ids', return_value={1, 2}):
            confirm(self.ws, case['id'], 'my confirmed API edit', {'id': 'c'*32}, 4)
        self.pending[0] = read_state(self.ws)['queue'][0]
        manifest, excluded = self.prepare()
        self.assertFalse(excluded)
        self.assertEqual(manifest['counts']['train'], 10)
        item = manifest['examples']['train'][0]
        self.assertEqual(item['source_kind'], 'owner_confirmed_reply')
        self.assertEqual(item['row']['completion'][0]['content'], 'my confirmed API edit')
        self.assertFalse(any(v.get('replay') for v in manifest['examples']['train']))


class ActivationTests(unittest.TestCase):
    def test_quality_regression_and_instruction_failure_block_promotion(self):
        from mybot.services.style_versions import quality_passed, quality_path
        from mybot.ai.local_format import FORMAT_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            version = 'a'*32
            root = ws.root / 'learning/mlx_runs' / version
            run = {'dataset_id': 'b'*32, 'adapter_sha256': 'weights', 'base_model': 'model',
                   'loss_before': 2., 'loss_after': 1.9, 'completed_steps': 10, 'steps': 10}
            write_json(root / 'run.json', run)
            write_json(ws.root / 'learning/reviewed' / ('b'*32 + '.json'), {'version': 4})
            report = {'status': 'passed', 'method': 'online_validation_and_instructions_v1',
                      'format_version': FORMAT_VERSION, 'account_id': 1, 'dialog_id': 2,
                      'adapter_sha256': 'weights', 'tokenizer_hash': 'tokens',
                      'cases': ['c1', 'c2'], 'finite_weights': True, 'validation_passed': True,
                      'results': [{'passed': True, 'instruction': ''}, {'passed': True, 'instruction': 'explicit'}]*2}
            with patch('mybot.services.history_training.tokenizer_hash', return_value='tokens'), patch(
                    'mybot.storage.style_lesson.StyleLesson.holdout_for', return_value={'cases': ['c1', 'c2']}):
                write_json(quality_path(ws, version), report)
                self.assertTrue(quality_passed(ws, version))
                run['loss_after'] = 2.1
                write_json(root / 'run.json', run)
                self.assertFalse(quality_passed(ws, version))
                run['loss_after'] = 1.9
                write_json(root / 'run.json', run)
                report['results'][1]['passed'] = False
                write_json(quality_path(ws, version), report)
                self.assertFalse(quality_passed(ws, version))

    def test_automatic_parent_compare_and_rollback(self):
        from mybot.services.style_versions import activate, rollback, selected
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            parent, candidate = 'a'*32, 'b'*32
            root = ws.root / 'learning/mlx_runs' / candidate
            write_json(root / 'run.json', {'purpose': 'online', 'parent_adapter': parent})
            with patch('mybot.services.style_versions.adapter', return_value=root):
                activate(ws, parent, explicit_owner=True)
                activate(ws, candidate, automatic=True)
                self.assertEqual(selected(ws), candidate)
                rollback(ws)
                self.assertEqual(selected(ws), parent)
                activate(ws, 'base')
                with self.assertRaises(ValueError):
                    activate(ws, candidate, automatic=True)


class GPUWaitTests(unittest.TestCase):
    def test_wait_preserves_checkpoint_progress_and_metadata(self):
        from scripts.online_reply_training import waiting_gpu
        from mybot.storage.atomic import read_json
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            root = ws.root / 'learning/mlx_runs' / ('a' * 32)
            write_json(root / 'run.json', {'completed_steps': 17, 'dataset_id': 'source',
                                         'cycle_elapsed_seconds': 321})
            write_json(root / 'checkpoint.json', {'position': 17})
            waiting_gpu(ws, root)
            state = read_json(root / 'run.json')
            self.assertEqual(state['completed_steps'], 17)
            self.assertEqual(state['dataset_id'], 'source')
            self.assertEqual(state['cycle_elapsed_seconds'], 321)
            self.assertEqual(state['status'], 'paused')
            self.assertEqual(read_json(root / 'checkpoint.json'), {'position': 17})

    def test_unstarted_wait_is_not_a_resumable_checkpoint(self):
        from scripts.online_reply_training import waiting_gpu
        from mybot.storage.atomic import read_json
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            root = ws.root / 'learning/mlx_runs' / ('a' * 32)
            waiting_gpu(ws, root)
            self.assertEqual(read_json(root / 'run.json')['status'], 'waiting_gpu')

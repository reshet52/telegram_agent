"""Offline source isolation, full coverage, formatting and overnight controls."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, AsyncMock

from mybot.services.full_history_dataset import prepare_full, fit, seal
from mybot.services.night_training_control import NightGuard, BudgetExpired
from mybot.services.reviewed_dataset import load_reviewed
from mybot.storage.atomic import write_json, read_json
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.workspace import Workspace


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(m['content'] for m in messages) + '<|im_start|>assistant\n'

    def encode(self, text, **kwargs):
        return list(map(ord, text))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'Synthetic', Path(self.temp.name))
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.messages = [dict(message_id=i + 1, sender='Other' if i % 2 == 0 else 'Я',
                              text=f'question {i}' if i % 2 == 0 else f'reply {i}', type='text',
                              date=(start + timedelta(hours=(i // 2) * 5, seconds=i % 2)).isoformat())
                         for i in range(1000)]
        self.save()

    def save(self):
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))

    def prepare(self):
        return prepare_full(self.ws, Tokenizer(), self.ws.root, max_length=512)

    def test_full_coverage_disjoint_holdout_schedule_and_append(self):
        data = self.prepare()
        self.assertEqual(data['counts'], {'train': 475, 'valid': 25})
        self.assertEqual(len(data['coverage']), 1000)
        self.assertTrue(all(r['uses'] or r['reason'] for r in data['coverage']))
        train_ids = {i for r in data['examples']['train'] for i in r['ids']}
        valid_ids = {i for r in data['examples']['valid'] for i in r['ids']}
        self.assertFalse(train_ids & valid_ids)
        order = data['order']
        self.assertEqual(set(order[:475]), set(range(475)))
        self.assertEqual(set(order[475:]), set(range(475 * 2 // 3, 475)))
        holdout = StyleLesson(self.ws).holdout_for(data['holdout_ref'])
        self.assertEqual(len(holdout['cases']), 20)
        used = set()
        for key in holdout['cases']:
            case = StyleLesson(self.ws).store.get(key)
            ids = set(case['context_ids'] + case['expected_ids'])
            self.assertFalse(ids & used)
            used |= ids
        self.assertFalse((self.ws.root / 'learning/holdout_active.json').exists())
        load_reviewed(self.ws, data['id'])
        self.messages.append(dict(self.messages[-1], message_id=1001, text='new'))
        self.save()
        load_reviewed(self.ws, data['id'])

    def test_source_changes_deletions_and_foreign_workspace_fail(self):
        data = self.prepare()
        other = Workspace(9, 2, 'Other', self.ws.root)
        with self.assertRaises(ValueError):
            load_reviewed(other, data['id'])
        write_json(self.ws.deleted_message_ids, [10])
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, data['id'])
        write_json(self.ws.deleted_message_ids, [])
        self.messages[0]['text'] = 'edited'
        self.save()
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, data['id'])

    def test_previous_training_never_enters_new_validation(self):
        root = self.ws.root / 'learning'
        write_json(root / 'reviewed' / ('a' * 32 + '.json'),
                   {'examples': {'train': [{'ids': [951, 952]}]}})
        write_json(root / 'mlx_runs/old/run.json', dict(account_id=1, dialog_id=2,
                   completed_steps=1, dataset_id='a' * 32))
        data = self.prepare()
        self.assertEqual(data['counts']['valid'], 24)
        self.assertTrue(all(not {951, 952}.intersection(r['ids']) for r in data['examples']['valid']))

    def test_media_and_deletions_are_barriers(self):
        self.messages[0]['type'] = 'photo'
        self.messages[4]['text'] = ''
        self.save()
        write_json(self.ws.deleted_message_ids, [3])
        data = self.prepare()
        ledger = {r['message_id']: r for r in data['coverage']}
        self.assertEqual(ledger[1]['reason'], 'media_without_verified_text')
        self.assertEqual(ledger[3]['reason'], 'deleted')
        self.assertEqual(ledger[5]['reason'], 'empty_text')
        self.assertFalse(any({1, 2, 3, 4, 5, 6}.intersection(r['ids']) for r in data['examples']['train']))

    def test_confirmed_replacement_supersedes_original_and_revocation_fails(self):
        lesson = StyleLesson(self.ws)
        case = lesson.store.create_from([self.messages[0]], [self.messages[1]])
        lesson.store.save_result(case['id'], ['first', 'second', 'third'])
        lesson.store.save_correction(case['id'], 'corrected reply')
        data = self.prepare()
        rows = data['examples']['train']
        self.assertTrue(any(r['row']['completion'][0]['content'] == 'corrected reply' for r in rows))
        self.assertFalse(any(r['case_id'] == 'full:2' for r in rows))
        load_reviewed(self.ws, data['id'])
        lesson.store.clear_learning(case['id'])
        with self.assertRaises(ValueError):
            load_reviewed(self.ws, data['id'])

    def test_trim_preserves_full_last_incoming_and_target(self):
        context = [{'sender': 'Я', 'text': 'old' * 500},
                   {'sender': 'Other', 'text': 'first incoming'},
                   {'sender': 'Other', 'text': 'second incoming'}]
        response = [{'sender': 'Я', 'text': 'my entire answer'}]
        kept, row = fit(context, response, Tokenizer(), 512)
        self.assertEqual(kept, context[1:])
        self.assertEqual(row['completion'][0]['content'], 'my entire answer')
        self.assertIsNone(fit(context, [{'text': 'x' * 1000}], Tokenizer(), 512))
        self.assertIsNone(fit([{'sender': 'Other', 'text': 'x' * 1000}], response, Tokenizer(), 512))


class FakeClock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


class GuardTests(unittest.TestCase):
    def test_thermal_pause_recovers_and_budget_includes_pause(self):
        clock = FakeClock()
        reader = lambda: 2 if clock() < 30 else 0
        guard = NightGuard(100, reserve=10, reader=reader, clock=clock, sleep=clock.sleep)
        guard.wait()
        self.assertEqual(guard.pause_seconds, 30)
        self.assertEqual(guard.pause_count, 1)
        clock.value = 90
        with self.assertRaises(BudgetExpired):
            guard.wait()

    def test_unknown_thermal_duty_cycle_and_cancel(self):
        clock = FakeClock()
        guard = NightGuard(10000, reserve=0, reader=lambda: None, clock=clock, sleep=clock.sleep)
        clock.value = 1200
        guard.wait()
        self.assertEqual(clock(), 1500)
        self.assertEqual(guard.pause_seconds, 300)
        guard.stopping = True
        with self.assertRaises(BudgetExpired):
            guard.wait()

    def test_critical_temperature_cannot_wait_past_deadline(self):
        clock = FakeClock()
        guard = NightGuard(100, reserve=20, reader=lambda: 3, clock=clock, sleep=clock.sleep)
        with self.assertRaises(BudgetExpired):
            guard.wait()
        self.assertEqual(clock(), 80)


class LocalContextTests(unittest.TestCase):
    def test_empty_live_context_equals_training_and_owner_settings_are_preserved(self):
        from mybot.ai.local_reply import live_messages
        from mybot.ai.local_context import build_local_context
        from mybot.services.learning_dataset import example
        from mybot.storage.dialog_goal import save_goal
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'Test', Path(tmp))
            recent = [{'sender': 'Other', 'text': 'coffee tomorrow?'}]
            from mybot.ai.reply_contract import DRAFT_POLICY
            expected = example(recent, [])['prompt']
            expected[0]['content'] += '\n\n' + DRAFT_POLICY
            self.assertEqual(live_messages('', recent), expected)
            save_goal(ws, 'Do not promise anything')
            write_json(ws.agent_memory, {'user_facts': [{'memory': 'Likes coffee'}]})
            context = build_local_context(recent, 'be brief', {}, ws)
            self.assertIn('Do not promise anything', context)
            self.assertIn('Likes coffee', context)
            messages = live_messages(context, recent)
            self.assertEqual(messages[-1]['content'], 'coffee tomorrow?')
            self.assertNotIn('ПОХОЖИЕ ИСТОРИЧЕСКИЕ СИТУАЦИИ', context)


class JobsTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_prepare_flags_and_resume_requires_checkpoint(self):
        from mybot.services import training_jobs
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'Test', Path(tmp))
            with patch.object(training_jobs, 'available', return_value=True), \
                 patch.object(training_jobs, 'spawn', AsyncMock(return_value={})) as spawn:
                await training_jobs.preview_history(ws, full=True)
                self.assertIn('--full', spawn.call_args.args[3])
            with self.assertRaises(ValueError):
                await training_jobs.resume_full(ws, 'a' * 32)

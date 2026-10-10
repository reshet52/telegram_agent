"""Offline end-to-end lesson, fixed holdout, blind comparison and real routing."""

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.storage.workspace import Workspace
from mybot.storage.atomic import write_json, read_json
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.reply_practice import PracticeStore
from mybot.services.reviewed_dataset import prepare_reviewed, load_reviewed
from mybot.services import style_versions, style_comparison, training_jobs
from mybot.services.reply_practice import generate_practice
from mybot.services.reply_service import ReplyService
from mybot.telegram.bot_style_lesson import BotStyleLesson


class StyleLessonTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_holdout_preserves_old_adapter_and_comparison_sources(self):
        from mybot.services.style_comparison import open_comparison
        version = self.trained()
        original = self.lesson.holdout()['cases']
        self.lesson.save_replacement_preview()
        replacement = self.lesson.replace_holdout()
        self.assertNotEqual(replacement['cases'], original)
        self.assertTrue(style_versions.quality_passed(self.ws, version))
        self.assertEqual(open_comparison(self.ws, version)['cases'], original)

    async def test_wait_confirmation_trains_action_and_reply_override(self):
        from mybot.services.reviewed_dataset import action_examples
        self.trained()
        api = NS(send_message=AsyncMock(return_value=NS(message_id=101)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda _: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='wait')), \
             patch('mybot.ai.local_reply.generate', AsyncMock()) as generate:
            await ui.handle('ui:sl:start:7:9', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            key = self.lesson.current()['cases'][0]
            self.assertEqual(self.lesson.store.get(key)['model_action']['action'], 'wait')
            generate.assert_not_awaited()
            await ui.handle(f'ui:sl:wait:{key}', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            self.assertEqual(self.lesson.current()['index'], 1)
            target = next(item for item in action_examples(self.ws) if item['case_id'] == 'action:' + key)
            self.assertEqual(target['row']['completion'][0]['content'], 'WAIT')

    async def test_lesson_issues_are_independent_and_need_a_positive_target(self):
        from mybot.services.practice_learning import chosen_reply
        self.trained()
        state = self.lesson.start()
        key = state['cases'][0]
        case = self.lesson.store.save_result(key, ['First text', 'Second text'])
        self.assertEqual(case['variant_count'], 2)
        case = self.lesson.store.flag_issue(key, 1, 'emoji')
        case = self.lesson.store.flag_issue(key, 2, 'facts')
        self.assertEqual(case['issues'], {'1': ['emoji'], '2': ['facts']})
        case = self.lesson.store.rate(key, 1, 'good')
        self.assertIsNone(chosen_reply(case, []))
        case = self.lesson.store.flag_issue(key, 1, 'emoji')
        self.assertEqual(chosen_reply(case, []), ('approved_suggestion_not_sent', 'First text'))
        case = self.lesson.store.save_correction(key, 'My corrected answer')
        self.assertEqual(chosen_reply(case, [])[0], 'owner_correction_not_sent')
        with self.assertRaises(ValueError):
            self.lesson.store.flag_issue(key, 3, 'facts')

    async def test_lesson_without_trained_model_never_generates_base(self):
        state = self.lesson.start()
        with patch('mybot.ai.local_reply.generate', AsyncMock()) as gen:
            with self.assertRaisesRegex(ValueError, 'Базовая модель'):
                await generate_practice(self.lesson.store, state['cases'][0], local=True)
        gen.assert_not_awaited()

    async def test_lesson_buttons_flag_reject_and_save_correction(self):
        self.trained()
        api = NS(send_message=AsyncMock(return_value=NS(message_id=101)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda _: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['First text', 'Second text'])):
            await ui.handle('ui:sl:start:7:9', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            key = self.lesson.current()['cases'][0]
            buttons = [b.text for row in api.edit_message_text.call_args.kwargs['reply_markup'].inline_keyboard for b in row]
            self.assertIn('Первый', buttons)
            self.assertIn('Второй', buttons)
            self.assertIn('1: Чужие эмодзи', buttons)
            await ui.handle(f'ui:sl:issue:{key}:1:emoji', None, None)
            await ui.handle(f'ui:sl:pick:{key}:1', None, None)
            self.assertEqual(self.lesson.current()['index'], 0)
            self.assertEqual(self.lesson.store.get(key)['ratings'], {})
            await ui.handle(f'ui:sl:neither:{key}', None, None)
            self.assertTrue(self.lesson.store.get(key)['learning_disabled'])
            await ui.handle(f'ui:sl:correct:{key}', None, None)
            await ui.handle_text('My corrected reply', None)
            await asyncio.gather(*list(ui.tasks.values()))
            self.assertEqual(self.lesson.current()['index'], 1)
            self.assertEqual(self.lesson.store.get(key)['owner_correction']['text'], 'My corrected reply')

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ws = Workspace(7, 9, 'Test', Path(temp.name))
        messages = [dict(message_id=i, sender='Я' if i % 2 == 0 else 'Other',
                         text=f'message-{i}', type='text') for i in range(1, 1001)]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
        self.lesson = StyleLesson(self.ws)

    async def reviewed(self):
        state = self.lesson.start()
        for key in state['cases']:
            self.lesson.store.save_result(key, [f'answer-{key}', 'second'])
            self.lesson.store.rate(key, 1, 'good')
            self.lesson.advance(key, 'pick')
        return prepare_reviewed(self.ws)

    def fake_adapter(self, manifest):
        version = 'a' * 32
        root = self.ws.root / 'learning/mlx_runs' / version
        root.mkdir(parents=True)
        (root / 'adapters.safetensors').write_bytes(b'synthetic adapter')
        write_json(root / 'run.json', {'account_id': 7, 'dialog_id': 9, 'status': 'trained_unapproved',
            'purpose': manifest.get('purpose', 'reviewed'),
            'dataset_id': manifest['id'], 'adapter_sha256': hashlib.sha256(b'synthetic adapter').hexdigest()})
        from mybot.ai.local_format import FORMAT_VERSION
        write_json(style_versions.quality_path(self.ws, version), dict(account_id=7, dialog_id=9,
            status='passed', format_version=FORMAT_VERSION, adapter_sha256=hashlib.sha256(b'synthetic adapter').hexdigest(),
            cases=self.lesson.holdout()['cases'], counts={'candidate': 10}, expected_per_model=10,
            valid_actions=10, expected_actions=10))
        return version

    def trained(self):
        from mybot.services.history_training import prepare
        from tests.test_history_training import Tokenizer
        version = self.fake_adapter(prepare(self.ws, Tokenizer(), self.ws.root, limit=30))
        style_versions.activate_practice(self.ws, version)
        return version

    async def test_fixed_holdout_resume_skip_and_stale_click(self):
        state = self.lesson.start()
        reserved = self.lesson.reserved_ids()
        for key in state['cases']:
            case = self.lesson.store.get(key)
            self.assertLess(max(case['context_ids'] + case['expected_ids']), min(reserved))
        key = state['cases'][0]
        self.lesson.advance(key, 'skip')
        with self.assertRaises(ValueError):
            self.lesson.advance(key, 'skip')
        self.lesson.pause()
        restarted = StyleLesson(self.ws)
        resumed = restarted.start()
        self.assertEqual(resumed['id'], state['id'])
        self.assertEqual(resumed['index'], 1)
        self.assertEqual(reserved, restarted.reserved_ids())
        with self.assertRaises(ValueError):
            StyleLesson(Workspace(8, 9, 'wrong', self.ws.root)).current()

    async def test_local_prompt_never_contains_hidden_reply_or_future_and_is_cached(self):
        self.trained()
        state = self.lesson.start()
        key = state['cases'][0]
        case = self.lesson.store.get(key)
        async def generate(request):
            serialized = json.dumps(request)
            for i in case['expected_ids']:
                self.assertNotIn(f'message-{i}"', serialized)
            self.assertEqual(request['context_message_ids'], case['context_ids'])
            request['receipt'] = {'adapter_sha256': None, 'seconds': 1, 'model': 'local'}
            self.assertEqual(request['count'], 2)
            return ['one', 'two']
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', side_effect=generate) as local, \
             patch('mybot.services.reply_practice.generate_answers', side_effect=AssertionError('cloud')):
            first = await generate_practice(self.lesson.store, key, local=True)
            second = await generate_practice(self.lesson.store, key, local=True)
        self.assertEqual(first, second)
        self.assertEqual(local.call_count, 1)
        self.assertEqual(first['request']['provider'], 'mlx')
        action_prompt = json.dumps(first['action_request']['messages'])
        for source_id in case['expected_ids']:
            self.assertNotIn(f'message-{source_id}"', action_prompt)

    async def test_holdout_never_becomes_training_target_and_revocation_invalidates_adapter(self):
        manifest = await self.reviewed()
        self.assertTrue(manifest['fixed_holdout'])
        _, rows = load_reviewed(self.ws, manifest['id'])
        self.assertEqual(len(rows['valid']), 5)
        version = self.fake_adapter(manifest)
        style_versions.adapter(self.ws, version)
        key = manifest['examples']['train'][0]['case_id']
        self.lesson.store.clear_learning(key)
        with self.assertRaises(ValueError):
            style_versions.adapter(self.ws, version)

    async def test_comparison_cache_votes_activation_routing_and_rollback(self):
        manifest = await self.reviewed()
        version = self.fake_adapter(manifest)
        with self.assertRaises(ValueError):
            style_versions.activate(self.ws, version)
        comparison = style_comparison.open_comparison(self.ws, version)
        async def local(request, **kwargs):
            request['receipt'] = {'adapter_sha256': 'receipt', 'seconds': .2, 'model': request['model']}
            return ['NEW' if request['version'] == version else 'OLD']
        with patch('mybot.ai.local_reply.generate', side_effect=local) as generate:
            for i in range(len(comparison['cases'])):
                pair = await style_comparison.generate_pair(self.ws, version, i)
                await style_comparison.generate_pair(self.ws, version, i)
                position = pair['orders'][str(i)].index('candidate')
                style_comparison.vote(self.ws, version, i, 'first' if position == 0 else 'second')
            self.assertEqual(generate.call_count, 10)
        style_versions.activate(self.ws, version)
        service = ReplyService([{'message_id': 1001, 'sender': 'Other', 'text': 'hello'}], {}, self.ws)
        async def three(request):
            self.assertEqual(request['version'], version)
            self.assertTrue(request['adapter'].endswith(version))
            request['receipt'] = {'adapter_sha256': 'receipt'}
            return ['one', 'two', 'three']
        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='current context')), \
             patch('mybot.ai.local_reply.generate', side_effect=three), \
             patch('mybot.services.reply_service.generate_answers', side_effect=AssertionError('cloud')):
            result = await service.generate()
        self.assertEqual(service.split_variants(result), ['one', 'two', 'three'])
        self.assertEqual(service.last_request_snapshot['version'], version)
        self.assertIn('receipt', read_json(self.ws.ai_request_preview))
        style_versions.rollback(self.ws)
        self.assertEqual(style_versions.selected(self.ws), 'configured')
        with self.assertRaises(ValueError):
            style_versions.adapter(Workspace(1, 9, 'wrong', self.ws.root), version)

    async def test_telegram_card_selection_progress_and_restart(self):
        self.trained()
        api = NS(send_message=AsyncMock(return_value=NS(message_id=55)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda u: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['a', 'b'])):
            await ui.handle('ui:sl:start:7:9', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            state = self.lesson.current()
            key = state['cases'][0]
            await ui.handle(f'ui:sl:pick:{key}:1', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            self.assertEqual(self.lesson.current()['index'], 1)
            restarted = BotStyleLesson(bot)
            await restarted.handle('ui:lesson', None, None)
            api.send_message.assert_awaited_once()
            self.assertGreater(api.edit_message_text.await_count, 2)
            self.assertIn('1/10', api.edit_message_text.await_args.kwargs['text'])
            for call in api.edit_message_text.await_args_list:
                self.assertLessEqual(len(call.kwargs['text'].encode('utf-16-le')) // 2, 4096)

    async def test_dead_training_job_and_cancel_identity(self):
        self.lesson.write('training_job.json', {'id': 'b' * 32, 'pid': 99999, 'state': 'training'})
        with patch('mybot.services.training_jobs.alive', return_value=False), \
             patch('mybot.services.training_jobs.os.kill') as kill:
            self.assertEqual(training_jobs.status(self.ws)['state'], 'failed')
            training_jobs.stop(self.ws)
            kill.assert_not_called()
        with patch('mybot.services.training_jobs.subprocess.run', return_value=NS(stdout='unrelated process')):
            self.assertFalse(training_jobs.alive({'id': 'b' * 32, 'pid': 99999}))

    async def test_deleted_lesson_source_can_be_skipped(self):
        from mybot.storage.deletions import mark_messages_deleted
        state = self.lesson.start()
        key = state['cases'][0]
        case = self.lesson.store.get(key)
        mark_messages_deleted({case['context_ids'][0]}, filename=self.ws.deleted_message_ids)
        with self.assertRaises(ValueError):
            self.lesson.require(key)
        self.lesson.advance(key, 'skip')
        self.assertEqual(self.lesson.current()['index'], 1)

    async def test_missing_local_model_never_falls_back(self):
        from mybot.ai import local_reply
        with patch('mybot.ai.local_reply.available', return_value=False), \
             patch('mybot.ai.local_reply.asyncio.create_subprocess_exec', AsyncMock()) as spawn:
            with self.assertRaisesRegex(ValueError, 'API автоматически не используется'):
                await local_reply.generate({'count': 3})
            spawn.assert_not_awaited()

    async def test_local_receipt_rejects_wrong_adapter_and_timeout_kills_worker(self):
        from mybot.ai import local_reply
        result = json.dumps({'variants': ['a', 'b', 'c'], 'adapter_sha256': 'wrong'}).encode()
        process = NS(returncode=0, communicate=AsyncMock(return_value=(result, b'')))
        with patch('mybot.ai.local_reply.available', return_value=True), \
             patch('mybot.ai.local_worker.worker.exchange', AsyncMock(return_value=json.loads(result))), \
             patch('mybot.ai.local_worker.worker.stop', AsyncMock()):
            with self.assertRaisesRegex(ValueError, 'не соответствует'):
                await local_reply.generate({'count': 3, 'expected_adapter_sha256': 'expected'})
        from unittest.mock import Mock
        process = NS(returncode=None, communicate=AsyncMock(side_effect=asyncio.TimeoutError),
                     terminate=Mock(), wait=AsyncMock(), kill=Mock())
        with patch('mybot.ai.local_reply.available', return_value=True), \
             patch('mybot.ai.local_worker.worker.exchange', AsyncMock(side_effect=asyncio.TimeoutError)), \
             patch('mybot.ai.local_worker.worker.stop', AsyncMock()) as stop:
            with self.assertRaisesRegex(ValueError, '3 минуты'):
                await local_reply.generate({'count': 3})
        stop.assert_awaited_once()

    async def test_background_training_start_deduplicates_and_stop_targets_job(self):
        from unittest.mock import Mock
        await self.reviewed()
        process = NS(pid=12345, returncode=None, wait=AsyncMock(return_value=0))
        with patch('mybot.services.training_jobs.available', return_value=True), \
             patch('mybot.services.training_jobs.asyncio.create_subprocess_exec', AsyncMock(return_value=process)) as spawn:
            job = await training_jobs.start(self.ws)
            again = await training_jobs.start(self.ws)
            self.assertEqual(job['id'], again['id'])
            spawn.assert_awaited_once()
            self.assertIn('--run-id', spawn.await_args.args)
            with patch('mybot.services.training_jobs.os.kill') as kill, \
                 patch('mybot.services.training_jobs.force_stop', AsyncMock()):
                stopped = training_jobs.stop(self.ws)
                self.assertEqual(stopped['state'], 'stopping')
                self.assertEqual(kill.call_args.args[0], process.pid)
            await asyncio.sleep(0)
        training_jobs._processes.pop(job['id'], None)

    async def test_telegram_stale_dialog_nonowner_and_correction_resume(self):
        self.trained()
        api = NS(send_message=AsyncMock(return_value=NS(message_id=56)), edit_message_text=AsyncMock())
        bot = NS(workspace=self.ws, owner_id=1, application=NS(bot=api), is_owner=lambda u: True,
                 goal=NS(workspace=lambda: self.ws), runtime_controller=NS(lock=asyncio.Lock()))
        ui = BotStyleLesson(bot)
        await ui.handle('ui:sl:start:8:9', None, None)
        self.assertIsNone(self.lesson.current())
        bot.is_owner = lambda u: False
        await ui.handle('ui:sl:start:7:9', None, None)
        self.assertIsNone(self.lesson.current())
        bot.is_owner = lambda u: True
        with patch('mybot.ai.local_reply.decide', AsyncMock(return_value='reply')), \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['a', 'b'])):
            await ui.handle('ui:sl:start:7:9', None, None)
            await asyncio.gather(*list(ui.tasks.values()))
            key = self.lesson.current()['cases'][0]
            await ui.handle(f'ui:sl:correct:{key}', None, None)
            restarted = BotStyleLesson(bot)
            self.assertTrue(await restarted.handle_text('My own reply', None))
            await asyncio.gather(*list(restarted.tasks.values()))
            self.assertEqual(self.lesson.current()['index'], 1)
            self.assertEqual(self.lesson.store.get(key)['owner_correction']['text'], 'My own reply')

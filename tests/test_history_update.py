"""Incremental updates publish together or leave the original workspace intact."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.services.history_import import commit_batch
from mybot.services.runtime_controller import RuntimeController
from mybot.storage.atomic import write_json
from mybot.storage.control_state import ControlState
from mybot.storage.history import load_all_messages
from mybot.storage.history_transaction import HistoryTransaction, MARKER, recover_history_update
from mybot.storage.workspace import create_workspace


class HistoryUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_reply_refresh_reads_telegram_tail_without_history_or_index_writes(self):
        self.controller.client.get_messages = AsyncMock(return_value=list(reversed(self.messages)))
        recent = self.controller.runtime.recent_messages
        before = self.snapshot()
        with patch('mybot.telegram.exporter.message_to_data', AsyncMock(side_effect=lambda item, *args: item)):
            async with self.controller.lock:
                await asyncio.wait_for(self.controller.refresh_reply_context(), timeout=1)
        self.assertIs(self.controller.runtime.recent_messages, recent)
        self.assertEqual([m['message_id'] for m in recent], [1, 2, 3, 4, 5])
        self.assertEqual(self.snapshot(), before)
        self.controller.client.get_messages.assert_awaited_once_with(22, limit=15)

    async def test_explicit_update_repairs_processed_gap_and_resets_idle_state(self):
        from mybot.storage.atomic import read_json
        existing = [self.messages[0], *self.messages[2:4]]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in existing))
        self.controller.runtime.raw_history[:] = existing
        self.controller.runtime.recent_messages[:] = existing
        self.controller.runtime.known_message_ids.discard(2)
        write_json(self.ws.episode_embeddings, {'episodes': [{'episode_id': 1, 'embedding': [99]}]})
        write_json(self.ws.root / 'learning' / 'idle_history.json',
                   {'status': 'recovery_required', 'verified_id': 1})
        with patch('mybot.services.history_update.fetch_messages_after',
                   AsyncMock(return_value=self.messages[1:])) as fetch, \
             patch('mybot.services.history_recovery.create_embeddings', side_effect=self.embeddings):
            await self.run_update()
        self.assertEqual(fetch.await_args.args[2], 1)
        self.assertEqual([m['message_id'] for m in load_all_messages(self.ws.chat_history)], [1, 2, 3, 4, 5])
        self.assertEqual([m['message_id'] for m in self.controller.runtime.recent_messages], [1, 2, 3, 4, 5])
        self.assertEqual(read_json(self.ws.episode_embeddings)['episodes'], [])
        self.assertEqual(read_json(self.ws.root / 'learning' / 'idle_history.json')['status'], 'synced')
        backups = list((self.ws.root / 'history_recovery_backups').glob('*/chat_history.jsonl'))
        self.assertEqual(len(backups), 1)
        self.assertEqual([m['message_id'] for m in load_all_messages(backups[0])], [1, 3, 4])

    async def test_repair_api_failure_leaves_original_files_and_blocked_state(self):
        existing = [self.messages[0], *self.messages[2:4]]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in existing))
        write_json(self.ws.root / 'learning' / 'idle_history.json',
                   {'status': 'recovery_required', 'verified_id': 1})
        before = self.snapshot()
        with patch('mybot.services.history_update.fetch_messages_after', AsyncMock(return_value=self.messages[1:])), \
             patch('mybot.services.history_recovery.create_embeddings', AsyncMock(side_effect=ValueError('unavailable'))), \
             self.assertLogs(level='ERROR'):
            await self.run_update()
        self.assertEqual(self.snapshot(), before)

    async def test_idle_state_resumes_once_and_new_message_replaces_timer(self):
        from datetime import datetime, timedelta, timezone
        from mybot.services.idle_history import IdleHistory
        from mybot.storage.atomic import read_json
        listener = NS(workspace=self.ws, selected_dialog=self.bot.agent_manager.selected_dialog,
            message_processing_lock=asyncio.Lock(), recent_messages=list(self.messages),
            known_message_ids={1, 2, 3, 4}, raw_history=list(self.messages), recent_messages_limit=15)
        self.controller.listener = listener
        idle = IdleHistory(self.controller)
        path = idle.path(self.ws)
        write_json(path, {'account_id': 11, 'dialog_id': 22, 'message_id': 4,
            'verified_id': 4, 'at': (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat(),
            'status': 'pending'})
        with patch.object(idle, '_sync', AsyncMock()) as sync:
            idle.schedule(listener, {}, resumed=True)
            await idle.task
            sync.assert_awaited_once()
            self.assertEqual(read_json(path)['status'], 'synced')
            idle.schedule(listener, {}, resumed=True)
            self.assertTrue(idle.task.done())
            idle.schedule(listener, {'message_id': 5})
            self.assertEqual(read_json(path)['message_id'], 5)
            idle.stop()

    async def test_out_of_order_repair_refuses_processed_episode(self):
        from mybot.services.history_import import merge_out_of_order_batch
        before = self.ws.chat_history.read_bytes()
        with self.assertRaisesRegex(ValueError, 'контрольной точки'):
            merge_out_of_order_batch(self.ws, [{'message_id': 1, 'sender': 'Other', 'text': 'missed'}])
        self.assertEqual(self.ws.chat_history.read_bytes(), before)

    async def test_idle_sync_ignores_messages_already_saved_after_stale_cursor(self):
        from mybot.services.idle_history import IdleHistory
        listener = NS(workspace=self.ws, selected_dialog=self.bot.agent_manager.selected_dialog,
            message_processing_lock=asyncio.Lock(), known_message_ids={1, 2, 3},
            raw_history=list(self.messages[:3]), recent_messages=list(self.messages[:3]),
            recent_messages_limit=15)
        self.controller.listener = listener
        state = {'verified_id': 2}
        before = self.ws.chat_history.read_bytes()
        with patch('mybot.services.idle_history.fetch_messages_after',
                   AsyncMock(return_value=self.messages[2:4])):
            await IdleHistory(self.controller)._sync(listener, state)
        self.assertEqual(self.ws.chat_history.read_bytes(), before)
        self.assertEqual(state['verified_id'], 4)

    async def test_processed_gap_stops_retries_and_preserves_history(self):
        from datetime import datetime, timedelta, timezone
        from mybot.services.idle_history import IdleHistory
        from mybot.storage.atomic import read_json
        listener = NS(workspace=self.ws, selected_dialog=self.bot.agent_manager.selected_dialog,
            message_processing_lock=asyncio.Lock(), known_message_ids={1, 2, 3, 4},
            raw_history=list(self.messages[:4]), recent_messages=list(self.messages[:4]),
            recent_messages_limit=15)
        self.controller.listener = listener
        idle = IdleHistory(self.controller)
        path = idle.path(self.ws)
        write_json(path, {'account_id': 11, 'dialog_id': 22, 'message_id': 4,
            'verified_id': 1, 'at': (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat(),
            'status': 'pending'})
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n'
                                                for m in (self.messages[0], *self.messages[2:4])))
        before = self.ws.chat_history.read_bytes()
        with patch('mybot.services.idle_history.fetch_messages_after',
                   AsyncMock(return_value=[self.messages[1]])) as fetch:
            idle.schedule(listener, {}, resumed=True)
            await idle.task
            self.assertEqual(read_json(path)['status'], 'recovery_required')
            idle.schedule(listener, {}, resumed=True)
            idle.schedule(listener, {'message_id': 5})
            self.assertEqual(fetch.await_count, 1)
        self.assertEqual(self.ws.chat_history.read_bytes(), before)
        self.assertEqual(read_json(path)['status'], 'recovery_required')

    async def test_idle_sync_repairs_missed_message_before_new_live_message(self):
        from mybot.services.idle_history import IdleHistory
        # Message 5 arrived live while message 4 was missed during disconnection.
        commit_batch(self.ws.chat_history, [self.messages[4]])
        listener = NS(workspace=self.ws, selected_dialog=self.bot.agent_manager.selected_dialog,
            message_processing_lock=asyncio.Lock(), known_message_ids={1, 2, 3, 4, 5},
            raw_history=list(self.messages), recent_messages=list(self.messages), recent_messages_limit=15)
        # Make 4 the missed message and keep its ID out of the listener.
        existing = [*self.messages[:3], self.messages[4]]
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in existing))
        listener.known_message_ids = {1, 2, 3, 5}
        self.controller.listener = listener
        state = {'verified_id': 3}
        old_episodes = self.ws.episodes.read_bytes()
        with patch('mybot.services.idle_history.fetch_messages_after',
                   AsyncMock(return_value=self.messages[3:])) as fetch:
            await IdleHistory(self.controller)._sync(listener, state)
        self.assertEqual(fetch.await_args.args[2], 3)
        self.assertEqual([m['message_id'] for m in load_all_messages(self.ws.chat_history)], [1, 2, 3, 4, 5])
        self.assertEqual(state['verified_id'], 5)
        self.assertEqual(self.ws.episodes.read_bytes(), old_episodes)
        self.assertEqual([m['message_id'] for m in listener.recent_messages], [1, 2, 3, 4, 5])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = patch('mybot.storage.workspace.DATA_ROOT', Path(temp.name))
        root.start()
        self.addCleanup(root.stop)
        self.ws = create_workspace(11, 22, 'Dialog')
        self.other = create_workspace(12, 22, 'Other account')
        self.messages = [{'message_id': i, 'sender': 'Other' if i % 2 else 'Я',
                          'text': f'Message {i}'} for i in range(1, 6)]
        commit_batch(self.ws.chat_history, self.messages[:4])
        self.ws.episodes.write_text(json.dumps({'episode_id': 1, 'incoming': self.messages[:1],
            'response': self.messages[1:2]}) + '\n')
        self.ws.episode_embeddings_live.write_text(json.dumps({'episode_id': 1, 'embedding': [1]}) + '\n')
        write_json(self.ws.agent_memory, {'person_facts': [{'memory': 'Keep this'}]})
        write_json(self.ws.memory_embeddings, {'memories': []})
        self.other.chat_history.write_text('other workspace unchanged')
        state = ControlState(11)
        state.set(active_dialog_id=22)
        self.bot = NS(control_state=state, application=None, workspace=self.ws,
                      agent_manager=NS(selected_dialog=NS(id=22, name='Dialog')))
        self.controller = RuntimeController(Mock(), NS(id=11), self.bot)
        self.controller.listener = NS(selected_dialog=self.bot.agent_manager.selected_dialog,
            message_processing_lock=asyncio.Lock(), episode_tracker=object(),
            unregister=AsyncMock(), register=Mock())
        self.controller.runtime = NS(workspace=self.ws, known_message_ids={1, 2, 3, 4},
            raw_history=list(self.messages[:4]), recent_messages=list(self.messages[:4]),
            episode_tracker=object())

    def snapshot(self):
        return {p.name: p.read_bytes() for p in self.ws.root.iterdir() if p.is_file()}

    async def embeddings(self, episodes, **kwargs):
        return [{'episode_id': item['episode_id'], 'embedding': [1, 2]} for item in episodes]

    async def run_update(self):
        self.assertTrue(await self.controller.update_history(AsyncMock()))
        await self.controller.update_task

    async def test_only_missing_messages_and_episodes_publish_and_repeat_is_idempotent(self):
        old_memory = self.ws.agent_memory.read_bytes()
        old_episode = self.ws.episodes.read_text().splitlines()[0]
        with patch('mybot.services.history_update.fetch_messages_after',
                   new=AsyncMock(return_value=self.messages[4:])) as fetch, \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=self.embeddings) as embed:
            await self.run_update()
            self.assertEqual(fetch.await_args.args[2], 4)
            self.assertEqual(embed.await_args.args[0][0]['episode_id'], 2)
            after = self.snapshot()
            await self.run_update()
            self.assertEqual(fetch.await_args.args[2], 5)
            self.assertEqual(embed.await_count, 1)
        self.assertEqual([m['message_id'] for m in load_all_messages(self.ws.chat_history)], [1, 2, 3, 4, 5])
        self.assertEqual(self.ws.episodes.read_text().splitlines()[0], old_episode)
        self.assertEqual(self.ws.agent_memory.read_bytes(), old_memory)
        self.assertEqual(self.snapshot(), after)
        self.assertEqual(self.other.chat_history.read_text(), 'other workspace unchanged')
        self.assertEqual(ControlState(11).get('active_dialog_id'), 22)
        self.controller.listener.unregister.assert_not_awaited()

    async def test_cancel_mid_embedding_leaves_all_files_and_runtime_unchanged(self):
        before = self.snapshot()
        runtime = self.controller.runtime
        tracker = runtime.episode_tracker
        entered = asyncio.Event()
        async def blocked(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        with patch('mybot.services.history_update.fetch_messages_after',
                   new=AsyncMock(return_value=self.messages[4:])), \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=blocked):
            await self.controller.update_history(AsyncMock())
            await asyncio.wait_for(entered.wait(), 2)
            self.assertEqual(self.snapshot(), before)
            self.assertTrue(self.controller.listener.message_processing_lock.locked())
            await asyncio.wait_for(self.controller.pause(), 2)
        self.assertEqual(self.snapshot(), before)
        self.assertIs(self.controller.runtime, runtime)
        self.assertIs(runtime.episode_tracker, tracker)
        self.assertEqual(runtime.known_message_ids, {1, 2, 3, 4})
        self.assertFalse(self.controller.listener.message_processing_lock.locked())
        self.assertFalse(list(self.ws.root.glob('.history-update-*')))
        self.assertFalse(self.controller.busy)

    async def test_failure_during_publication_rolls_back_history_and_episodes(self):
        from mybot.storage import history_transaction
        before = self.snapshot()
        original = history_transaction.copy_atomic
        def fail_once(source, target):
            if source.parent.name == 'after' and target == self.ws.episodes:
                raise OSError('disk failure')
            original(source, target)
        with patch('mybot.services.history_update.fetch_messages_after',
                   new=AsyncMock(return_value=self.messages[4:])), \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=self.embeddings), \
             patch('mybot.storage.history_transaction.copy_atomic', side_effect=fail_once):
            with self.assertLogs(level='ERROR'):
                await self.run_update()
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.controller.update_committed)
        self.assertEqual(self.controller.runtime.known_message_ids, {1, 2, 3, 4})

    def test_restart_recovers_partial_publication_from_undo_journal(self):
        before = self.snapshot()
        transaction = HistoryTransaction(self.ws)
        write_json(self.ws.root / MARKER, {'directory': transaction.directory.name,
                                         'original': transaction.original})
        commit_batch(self.ws.chat_history, self.messages[4:])
        recover_history_update(self.ws)
        self.assertEqual(self.snapshot(), before)

    async def test_cancel_after_publication_does_not_claim_rollback(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def report(text, done=False):
            if done:
                entered.set()
                await release.wait()
        with patch('mybot.services.history_update.fetch_messages_after',
                   new=AsyncMock(return_value=self.messages[4:])), \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=self.embeddings):
            await self.controller.update_history(report)
            await asyncio.wait_for(entered.wait(), 2)
            await self.controller.pause()
            self.assertTrue(self.controller.update_committed)
            self.assertNotIn('отменено', self.controller.last_status)
            release.set()
            await self.controller.update_task

    async def test_missing_checkpoint_does_not_call_embedding_or_change_files(self):
        self.ws.episodes.write_text(json.dumps({'episode_id': 1, 'response': [{'message_id': 999}]}) + '\n')
        before = self.snapshot()
        with patch('mybot.services.history_update.fetch_messages_after', new=AsyncMock(return_value=[])), \
             patch('mybot.episodes.incremental.create_embeddings', new_callable=AsyncMock) as embed:
            with self.assertLogs(level='ERROR'):
                await self.run_update()
            embed.assert_not_awaited()
        self.assertEqual(self.snapshot(), before)

    async def test_updates_episodes_for_messages_already_saved_by_listener(self):
        commit_batch(self.ws.chat_history, self.messages[4:])
        before = self.ws.chat_history.read_bytes()
        with patch('mybot.services.history_update.fetch_messages_after', new=AsyncMock(return_value=[])), \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=self.embeddings) as embed:
            await self.run_update()
            embed.assert_awaited_once()
        self.assertEqual(self.ws.chat_history.read_bytes(), before)
        self.assertEqual(len(self.ws.episodes.read_text().splitlines()), 2)

    async def test_rollback_failure_disables_listener_and_retains_recoverable_backup(self):
        from mybot.storage import history_transaction
        before = self.snapshot()
        original = history_transaction.copy_atomic
        listener = self.controller.listener
        def unavailable(source, target):
            if target.parent == self.ws.root and (
                    source.parent.name == 'before' or target == self.ws.episodes):
                raise OSError('disk unavailable')
            original(source, target)
        with patch('mybot.services.history_update.fetch_messages_after',
                   new=AsyncMock(return_value=self.messages[4:])), \
             patch('mybot.episodes.incremental.create_embeddings', side_effect=self.embeddings), \
             patch('mybot.storage.history_transaction.copy_atomic', side_effect=unavailable):
            with self.assertLogs(level='ERROR'):
                await self.run_update()
        listener.unregister.assert_awaited_once()
        self.assertIsNone(self.controller.runtime)
        self.assertTrue((self.ws.root / MARKER).exists())
        recover_history_update(self.ws)
        self.assertEqual(self.snapshot(), before)

    async def test_cancel_before_start_does_not_touch_files_or_active_dialog(self):
        before = self.snapshot()
        runtime = self.controller.runtime
        await self.controller.update_history(AsyncMock())
        await self.controller.pause()
        self.assertEqual(self.snapshot(), before)
        self.assertIs(self.controller.runtime, runtime)
        self.assertFalse(self.controller.busy)

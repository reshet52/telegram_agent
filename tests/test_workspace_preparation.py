"""Preparation regression tests use temporary workspaces and fake AI/Telegram."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.storage.workspace import create_workspace, get_global_root
from mybot.storage.atomic import read_json, write_json
from mybot.storage.history import load_all_messages
from mybot.services.history_import import import_history, commit_batch
from mybot.services.workspace_initializer import WorkspaceInitializer, CATEGORIES, bounded_chunks
from mybot.services.global_style import load_global_style, validate_traits
from mybot.episodes.checkpoint import commit_batch as commit_episodes, recover_batch
from mybot.services.runtime_controller import RuntimeController
from mybot.ai.context_builder import build_ai_request


STYLE = {'length': 'short', 'formality': 'informal', 'emoji': 'moderate',
         'punctuation': 'minimal', 'message_splitting': 'multiple'}


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        patcher = patch('mybot.storage.workspace.DATA_ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ws = create_workspace(101, 201, 'First')

    def history(self, count=205, marker='local_private_fact'):
        records = [{'message_id': i, 'sender': 'Я' if i % 2 else 'Other',
                    'text': f'{marker} {i}', 'type': 'text'} for i in range(1, count + 1)]
        commit_batch(self.ws.chat_history, records)
        return records

    def ai_patches(self, analyze=None):
        analysis = {category: [] for category in CATEGORIES}
        analysis['person_facts'] = [{'fact': 'local_private_fact', 'confidence': 'high'}]
        async def merge(category, items):
            return json.dumps([{'memory': 'local_private_fact', 'confidence': 'high',
                                'evidence': 'repeated', 'time_context': 'stable'}])
        async def embeddings(items):
            return [dict(item, embedding=[0.1, 0.2]) for item in items]
        return (
            patch('mybot.services.workspace_initializer.analyze_full_history',
                  new=analyze or AsyncMock(return_value=json.dumps(analysis))),
            patch('mybot.services.workspace_initializer.merge_memory_category', side_effect=merge),
            patch('mybot.services.workspace_initializer.create_memory_embeddings', side_effect=embeddings),
            patch('mybot.services.workspace_initializer.ai_client.responses.create',
                  new=AsyncMock(return_value=NS(output_text=json.dumps(STYLE)))),
        )

    async def test_resume_analysis_and_do_not_rebill_cached_parts(self):
        self.history()
        analysis = {category: [] for category in CATEGORIES}
        analysis['person_facts'] = [{'fact': 'local_private_fact'}]
        failing = AsyncMock(side_effect=[json.dumps(analysis), RuntimeError('network stopped')])
        patches = self.ai_patches(failing)
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaisesRegex(RuntimeError, 'network stopped'):
                await WorkspaceInitializer(self.ws, AsyncMock()).run()
        self.assertEqual(read_json(self.ws.root / 'initialization_state.json')['status'], 'failed')
        patches = self.ai_patches()
        with patches[0] as analysis_call, patches[1] as merge, patches[2] as embed, patches[3] as style:
            await WorkspaceInitializer(self.ws, AsyncMock()).run()
            self.assertEqual(analysis_call.await_count, 2)  # First of three chunks is reused.
            counts = [operation.await_count for operation in (analysis_call, merge, embed, style)]
            await WorkspaceInitializer(self.ws, AsyncMock()).run()
            self.assertEqual(counts, [operation.await_count for operation in (analysis_call, merge, embed, style)])
        self.assertEqual(read_json(self.ws.root / 'initialization_state.json')['status'], 'ready')
        self.assertEqual(read_json(self.ws.memory_update_state)['last_processed_message_id'], 205)
        self.assertTrue(read_json(self.ws.memory_embeddings)['memories'])
        self.assertTrue(self.ws.full_history_analysis.exists())
        self.assertEqual(load_global_style(101), STYLE)
        self.assertEqual(load_global_style(102), {})
        global_text = (get_global_root(101) / 'user_style.json').read_text()
        self.assertNotIn('local_private_fact', global_text)
        self.assertNotIn('local_private_fact', (get_global_root(101) / 'style_sources.json').read_text())
        second = create_workspace(101, 202, 'Second')
        self.assertFalse(second.agent_memory.exists())
        self.assertFalse(second.chat_history.exists())

    async def test_style_rejects_free_text_and_facts(self):
        with self.assertRaises(ValueError):
            validate_traits(dict(STYLE, name='someone'))
        with self.assertRaises(ValueError):
            validate_traits(dict(STYLE, formality='knows a private person'))
        self.history(4)
        patches = self.ai_patches()
        with patches[0], patches[1], patches[2], patches[3] as request:
            await WorkspaceInitializer(self.ws, AsyncMock()).run()
            inputs = json.loads(request.call_args.kwargs['input'])
            self.assertEqual(inputs, ['local_private_fact 1', 'local_private_fact 3'])

    async def test_history_restart_from_last_complete_batch(self):
        seen = []
        async def interrupted(dialog_id, min_id, **kwargs):
            seen.append(min_id)
            for i in range(min_id + 1, 206):
                if i == 204:
                    raise RuntimeError('interrupted')
                yield NS(id=i)
        async def complete(dialog_id, min_id, **kwargs):
            seen.append(min_id)
            for i in range(min_id + 1, 206):
                yield NS(id=i)
        async def convert(item, *args):
            return {'message_id': item.id, 'sender': 'Я', 'text': str(item.id)}
        with patch('mybot.services.history_import.message_to_data', side_effect=convert):
            with self.assertRaises(RuntimeError):
                await import_history(NS(iter_messages=interrupted), NS(id=201, name='First'),
                                     101, self.ws, AsyncMock())
            await import_history(NS(iter_messages=complete), NS(id=201, name='First'),
                                 101, self.ws, AsyncMock())
        records = load_all_messages(self.ws.chat_history, include_deleted=True)
        self.assertEqual(seen, [0, 200])
        self.assertEqual([item['message_id'] for item in records], list(range(1, 206)))

    def test_episode_publish_recovers_between_two_files(self):
        episodes = [{'episode_id': 1, 'incoming': [], 'response': []}]
        embeddings = [{'episode_id': 1, 'embedding': [1.0]}]
        from mybot.storage.atomic import write_text
        def interrupted(path, text):
            if Path(path) == self.ws.episodes:
                raise OSError('disk error')
            write_text(path, text)
        with patch('mybot.episodes.checkpoint.write_text', side_effect=interrupted):
            with self.assertRaises(OSError):
                commit_episodes(episodes, embeddings, self.ws.episodes, self.ws.episode_embeddings_live)
        recover_batch(self.ws.episodes, self.ws.episode_embeddings_live)
        recover_batch(self.ws.episodes, self.ws.episode_embeddings_live)
        self.assertEqual(len(self.ws.episodes.read_text().splitlines()), 1)
        self.assertEqual(len(self.ws.episode_embeddings_live.read_text().splitlines()), 1)

    async def test_pause_before_task_starts_does_not_leave_listener_active(self):
        old = NS(unregister=AsyncMock())
        bot = NS(agent_manager=NS(find_dialog=AsyncMock()), workspace=object(),
                 reply_service=object(), recent_messages=[])
        controller = RuntimeController(Mock(), NS(id=101), bot)
        controller.listener = old
        await controller.activate(201, AsyncMock())
        await controller.pause()
        self.assertFalse(controller.busy)
        self.assertEqual(controller.job['status'], 'paused')
        self.assertIsNone(controller.listener)
        old.unregister.assert_awaited_once()
        restarted = RuntimeController(Mock(), NS(id=101), bot)
        self.assertEqual(restarted.job['dialog_id'], 201)
        self.assertEqual(restarted.job['status'], 'paused')

    async def test_prompt_loads_only_account_style_and_current_workspace(self):
        self.history(2)
        patches = self.ai_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            await WorkspaceInitializer(self.ws, AsyncMock()).run()
        second = create_workspace(101, 202, 'Second')
        with patch('mybot.ai.context_builder.find_relevant_memories', new_callable=AsyncMock, return_value=[]) as search, \
             patch('mybot.ai.context_builder.find_similar_episodes', new_callable=AsyncMock, return_value=[]):
            prompt = await build_ai_request([], 'answer', {}, workspace=second)
        self.assertIn('informal', prompt)
        self.assertNotIn('local_private_fact', prompt)
        self.assertEqual(search.call_args.kwargs['base_embeddings_filename'], second.memory_embeddings)

    async def test_new_runtime_runs_full_pipeline_and_legacy_does_not(self):
        from mybot.services.dialog_runtime import prepare_dialog_runtime
        dialog, me = NS(id=201, name='First'), NS(id=101)
        self.history(2)
        async def no_import(*args):
            pass
        with patch('mybot.services.dialog_runtime.import_history', side_effect=no_import), \
             patch('mybot.services.dialog_runtime.check_recent_deletions', new_callable=AsyncMock, return_value=[]), \
             patch('mybot.services.dialog_runtime.initialize_episode_tracker', new_callable=AsyncMock,
                   return_value=(object(), [])), \
             patch('mybot.services.dialog_runtime.ensure_episode_embeddings', new_callable=AsyncMock), \
             patch('mybot.services.dialog_runtime.WorkspaceInitializer') as initializer:
            initializer.return_value.run = AsyncMock()
            runtime = await prepare_dialog_runtime(Mock(), dialog, me, NS(), progress=AsyncMock())
            initializer.return_value.run.assert_awaited_once()
            self.assertEqual(runtime.workspace.root, self.ws.root)
            write_json(self.ws.agent_memory, {})
            write_json(self.ws.memory_embeddings, {'memories': []})
            initializer.return_value.run.reset_mock()
            await prepare_dialog_runtime(Mock(), dialog, me, NS(), progress=AsyncMock())
            initializer.return_value.run.assert_not_awaited()
            await prepare_dialog_runtime(Mock(), dialog, me, NS(), progress=AsyncMock(), full_analysis=True)
            initializer.return_value.run.assert_awaited_once()

    async def test_repair_missing_embeddings_preserves_episode_ids(self):
        from mybot.episodes.maintenance import ensure_episode_embeddings
        from mybot.storage.atomic import write_text
        records = [{'episode_id': 5, 'incoming': []}, {'episode_id': 6, 'incoming': []}]
        write_text(self.ws.episodes, ''.join(json.dumps(item) + '\n' for item in records))
        write_json(self.ws.episode_embeddings, {'episodes': [{'episode_id': 5, 'embedding': [1.0]}]})
        with patch('mybot.episodes.maintenance.create_embeddings', new_callable=AsyncMock,
                   return_value=[{'episode_id': 6, 'embedding': [2.0]}]) as create:
            await ensure_episode_embeddings(self.ws, AsyncMock())
            await ensure_episode_embeddings(self.ws, AsyncMock())
            create.assert_awaited_once()
            self.assertEqual(create.call_args.args[0], [records[1]])
        self.assertEqual([json.loads(line)['episode_id'] for line in self.ws.episodes.read_text().splitlines()], [5, 6])

    def test_long_messages_are_split_without_dropping_text(self):
        text = 'x' * 45000
        chunks = list(bounded_chunks([{'text': text}]))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(''.join(item['text'] for chunk in chunks for item in chunk), text)


if __name__ == '__main__':
    unittest.main()

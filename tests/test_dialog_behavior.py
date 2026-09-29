"""Offline persistence, account isolation and Telegram settings tests."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.storage.workspace import Workspace
from mybot.storage.dialog_behavior import load_behavior, save_behavior_field
from mybot.telegram.bot_behavior import BotBehavior
from mybot.telegram.bot_goal import BotGoal
from mybot.ai.context_builder import build_ai_request


class BehaviorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ws = Workspace(1, 10, 'Test', Path(temp.name) / 'one')
        self.other = Workspace(2, 10, 'Other', Path(temp.name) / 'two')
        self.bot = NS(workspace=self.ws, is_owner=lambda update: True,
                      runtime_controller=NS(busy=False, lock=asyncio.Lock()),
                      callbacks=NS(show=AsyncMock()))
        self.bot.goal = BotGoal(self.bot)
        self.behavior = BotBehavior(self.bot)
        self.context = NS(user_data={})

    async def test_persistence_validation_and_isolation(self):
        save_behavior_field(self.ws, 'directness', 'direct')
        save_behavior_field(self.ws, 'boundaries', 'Без обещаний')
        self.assertEqual(load_behavior(self.ws)['directness'], 'direct')
        self.assertEqual(load_behavior(self.other)['boundaries'], '')
        for field, value in [('unknown', 'x'), ('directness', 'hostile'), ('current_state', 'x' * 801)]:
            with self.assertRaises(ValueError):
                save_behavior_field(self.ws, field, value)
        with patch('mybot.storage.atomic.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                save_behavior_field(self.ws, 'boundaries', 'new')
        self.assertEqual(load_behavior(self.ws)['boundaries'], 'Без обещаний')
        wrong = Workspace(2, 10, 'Other', self.ws.root)
        with self.assertRaises(ValueError):
            save_behavior_field(wrong, 'boundaries', 'overwrite')

    async def test_buttons_input_cancel_and_cross_account_staleness(self):
        await self.behavior.handle('ui:bp:1:10:d:1', None, self.context)
        self.assertEqual(load_behavior(self.ws)['directness'], 'direct')
        await self.behavior.handle('ui:bp:1:10:c:edit', None, self.context)
        self.assertTrue(await self.behavior.handle_text('Занят сегодня', None, self.context))
        self.assertEqual(load_behavior(self.ws)['current_state'], 'Занят сегодня')
        await self.behavior.handle('ui:bp:1:10:c:edit', None, self.context)
        await self.behavior.handle('ui:behavior', None, self.context)
        self.assertNotIn('awaiting_behavior', self.context.user_data)
        await self.behavior.handle('ui:bp:1:10:b:edit', None, self.context)
        self.bot.workspace = self.other
        await self.behavior.handle_text('Wrong account', None, self.context)
        await self.behavior.handle('ui:bp:1:10:d:1', None, self.context)
        self.assertEqual(load_behavior(self.other)['directness'], 'natural')
        self.assertEqual(load_behavior(self.other)['boundaries'], '')

    async def test_owner_busy_and_clear(self):
        self.bot.is_owner = lambda update: False
        await self.behavior.handle('ui:bp:1:10:d:1', None, self.context)
        self.assertEqual(load_behavior(self.ws)['directness'], 'natural')
        self.bot.is_owner = lambda update: True
        self.bot.runtime_controller.busy = True
        await self.behavior.handle('ui:bp:1:10:d:1', None, self.context)
        self.assertEqual(load_behavior(self.ws)['directness'], 'natural')
        self.bot.runtime_controller.busy = False
        save_behavior_field(self.ws, 'current_state', 'Temporary')
        await self.behavior.handle('ui:bp:1:10:c:clear', None, self.context)
        self.assertEqual(load_behavior(self.ws)['current_state'], '')

    async def test_prompt_only_contains_current_workspace_and_rereads_changes(self):
        save_behavior_field(self.ws, 'boundaries', 'UNIQUE_BOUNDARY')
        save_behavior_field(self.other, 'current_state', 'PRIVATE_OTHER')
        with patch('mybot.ai.context_builder.load_global_style', return_value={}), \
             patch('mybot.ai.context_builder.find_relevant_memories', AsyncMock(return_value=[])), \
             patch('mybot.ai.context_builder.find_similar_episodes', AsyncMock(return_value=[])):
            prompt = await build_ai_request([], 'Reply', {}, workspace=self.ws)
            self.assertIn('UNIQUE_BOUNDARY', prompt)
            self.assertNotIn('PRIVATE_OTHER', prompt)
            save_behavior_field(self.ws, 'boundaries', '')
            self.assertNotIn('UNIQUE_BOUNDARY', await build_ai_request([], 'Reply', {}, workspace=self.ws))

    async def test_callback_size_with_large_ids(self):
        ws = Workspace(9223372036854775807, -1001234567890, 'Test', self.ws.root)
        await self.behavior.overview(ws)
        markup = self.bot.callbacks.show.await_args.args[2]
        for row in markup.inline_keyboard:
            for button in row:
                self.assertLessEqual(len(button.callback_data.encode()), 64)

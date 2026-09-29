"""Persistent dialog instructions: entirely offline tests."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.storage.workspace import Workspace
from mybot.storage.dialog_goal import load_goal, save_goal, clear_goal
from mybot.telegram.bot_goal import BotGoal
from mybot.ai.context_builder import build_ai_request


class GoalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ws = self.workspace(1, 10)
        self.bot = NS(workspace=self.ws, is_owner=lambda update: True,
                      runtime_controller=NS(busy=False, lock=asyncio.Lock()),
                      callbacks=NS(show=AsyncMock()))
        self.goal = BotGoal(self.bot)
        self.context = NS(user_data={})

    def workspace(self, account, dialog):
        return Workspace(account, dialog, 'Test', self.root / str(account) / str(dialog))

    async def test_restart_isolation_and_clear(self):
        save_goal(self.ws, 'Тепло, без обещаний')
        self.assertEqual(load_goal(self.workspace(1, 10)), 'Тепло, без обещаний')
        self.assertEqual(load_goal(self.workspace(2, 10)), '')
        self.assertEqual(load_goal(self.workspace(1, 20)), '')
        clear_goal(self.ws)
        self.assertEqual(load_goal(self.workspace(1, 10)), '')

    async def test_atomic_failure_and_invalid_input_preserve_goal(self):
        save_goal(self.ws, 'Прежняя цель')
        with patch('mybot.storage.atomic.os.replace', side_effect=OSError('disk')):
            with self.assertRaises(OSError):
                save_goal(self.ws, 'Новая цель')
        for text in ('', ' ' * 10, 'x' * 2501):
            with self.assertRaises(ValueError):
                save_goal(self.ws, text)
        self.assertEqual(load_goal(self.ws), 'Прежняя цель')
        self.assertEqual([p.name for p in self.ws.root.iterdir()], ['dialog_goal.json'])

    async def test_wrong_owner_file_is_rejected(self):
        save_goal(self.ws, 'Прежняя цель')
        wrong = Workspace(2, 10, 'Other', self.ws.root)
        with self.assertRaises(ValueError):
            load_goal(wrong)
        with self.assertRaises(ValueError):
            clear_goal(wrong)
        self.assertEqual(load_goal(self.ws), 'Прежняя цель')

    async def test_edit_cancel_save_and_stale_button(self):
        save_goal(self.ws, 'Старая')
        await self.goal.handle('ui:goal:set:10', None, self.context)
        await self.goal.handle('ui:goal', None, self.context)
        self.assertNotIn('awaiting_goal', self.context.user_data)
        self.assertEqual(load_goal(self.ws), 'Старая')
        await self.goal.handle('ui:goal:set:10', None, self.context)
        self.assertTrue(await self.goal.handle_text('Новая', None, self.context))
        self.assertEqual(load_goal(self.ws), 'Новая')
        await self.goal.handle('ui:goal:clear:20', None, self.context)
        self.assertEqual(load_goal(self.ws), 'Новая')

    async def test_switch_during_input_does_not_write_other_dialog(self):
        await self.goal.handle('ui:goal:set:10', None, self.context)
        self.bot.workspace = self.workspace(2, 10)
        await self.goal.handle_text('Не туда', None, self.context)
        self.assertEqual(load_goal(self.bot.workspace), '')
        self.assertNotIn('awaiting_goal', self.context.user_data)

    async def test_owner_and_busy_guards(self):
        self.bot.is_owner = lambda update: False
        await self.goal.handle('ui:goal:write', None, self.context, text='Нет')
        self.bot.is_owner = lambda update: True
        self.bot.runtime_controller.busy = True
        await self.goal.handle('ui:goal:write', None, self.context, text='Нет')
        self.assertEqual(load_goal(self.ws), '')

    async def test_prompt_rereads_goal_and_never_uses_other_workspace(self):
        with patch('mybot.ai.context_builder.load_global_style', return_value={}), \
             patch('mybot.ai.context_builder.find_relevant_memories', new=AsyncMock(return_value=[])), \
             patch('mybot.ai.context_builder.find_similar_episodes', new=AsyncMock(return_value=[])):
            async def prompt(ws):
                return await build_ai_request([], 'Ответь естественно', {}, workspace=ws,
                                              use_correction_examples=False)
            save_goal(self.ws, 'UNIQUE_GOAL_FIRST')
            self.assertIn('UNIQUE_GOAL_FIRST', await prompt(self.ws))
            self.assertNotIn('UNIQUE_GOAL_FIRST', await prompt(self.workspace(2, 10)))
            save_goal(self.ws, 'UNIQUE_GOAL_NEXT')
            result = await prompt(self.ws)
            self.assertIn('UNIQUE_GOAL_NEXT', result)
            self.assertNotIn('UNIQUE_GOAL_FIRST', result)
            clear_goal(self.ws)
            self.assertNotIn('UNIQUE_GOAL_NEXT', await prompt(self.ws))

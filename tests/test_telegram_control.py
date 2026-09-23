"""Offline UI/lifecycle regressions; no Telegram or OpenAI requests."""

import asyncio
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from telegram.error import BadRequest
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_keyboards import main_menu, dialogs_menu
from mybot.telegram.bot_interface import BotInterface
from mybot.app.session_state import SessionState
from mybot.services.runtime_controller import RuntimeController
from mybot.telegram.live_events import LiveEvents


def message(number=1, keyboard=None):
    return NS(message_id=number, reply_markup=keyboard, edit_text=AsyncMock(),
              reply_text=AsyncMock(), delete=AsyncMock(), text="спокойное")


def update(source=None, data="ui:home", owner=1):
    return NS(effective_message=source or message(),
              effective_user=NS(id=owner), effective_chat=NS(id=owner, type="private"),
              callback_query=NS(data=data, answer=AsyncMock()))


class ControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        patcher = patch('mybot.services.runtime_controller.get_global_root',
                        return_value=Path(self.temporary.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def bot(self):
        bot = BotInterface(1, [], None, SessionState(), telegram_client=Mock())
        bot.menu = NS(show=AsyncMock(), move_to_bottom=AsyncMock())
        bot.control_bot_id = 999
        return bot

    async def test_menu_reuses_message_and_handles_repeat(self):
        sent = message()
        sender = AsyncMock(return_value=sent)
        menu = BotMenu(NS(owner_id=1, application=NS(bot=NS(send_message=sender))))
        await menu.show(None, 'home', main_menu())
        await menu.show(None, 'dialogs', main_menu())
        sender.assert_awaited_once()
        sent.edit_text.assert_awaited_once()
        sent.edit_text.side_effect = BadRequest('Message is not modified')
        await menu.show(None, 'dialogs', main_menu())
        self.assertEqual(sender.await_count, 1)
        sent.edit_text.side_effect = BadRequest('Message to edit not found')
        await menu.show(None, 'home', main_menu())
        self.assertEqual(sender.await_count, 2)

    async def test_cleanup_only_bot_owned_ui_messages(self):
        keyboard = NS(rows=[NS(buttons=[NS(data=b"ui:home")])])
        items = [NS(id=1, sender_id=99, reply_markup=keyboard),
                 NS(id=2, sender_id=99, reply_markup=None),
                 NS(id=3, sender_id=1, reply_markup=keyboard),
                 NS(id=4, sender_id=99, reply_markup=NS(rows=[NS(buttons=[NS(data=b"other")])]))]
        async def history(*args, **kwargs):
            for item in items:
                yield item
        api = NS(id=99, username='control', delete_message=AsyncMock())
        menu = BotMenu(NS(owner_id=1, application=NS(bot=api),
                          telegram_client=NS(iter_messages=history)))
        await menu.clean_previous_menus()
        api.delete_message.assert_awaited_once_with(chat_id=1, message_id=1)

    async def test_stale_menu_removed_but_content_untouched(self):
        menu = BotMenu(NS(owner_id=1, application=NS(bot=NS(send_message=AsyncMock()))))
        menu.message = message(10)
        old = message(2, main_menu())
        await menu.show(old, 'home', main_menu())
        old.delete.assert_awaited_once()
        content = message(3)
        await menu.show(content, 'home', main_menu())
        content.delete.assert_not_awaited()

    async def test_no_runtime_blocks_ai_and_memory(self):
        bot = self.bot()
        await bot.send_generated_answers(update())
        await bot.update_memory_command(update(), NS())
        self.assertEqual(bot.menu.show.await_count, 2)

    async def test_panel_button_is_navigation_not_ai_instruction(self):
        bot = self.bot()
        bot.send_generated_answers = AsyncMock()
        bot.callbacks.dispatch = AsyncMock()
        source = message()
        source.text = '💬 Диалоги'
        context = NS(user_data={})
        await bot.text_instruction(update(source), context)
        bot.callbacks.dispatch.assert_awaited_once()
        self.assertEqual(bot.callbacks.dispatch.call_args.args[0], 'ui:dialogs:0')
        bot.send_generated_answers.assert_not_awaited()
        source.delete.assert_awaited_once()

    async def test_owner_and_confirmation_required(self):
        bot = self.bot()
        bot.runtime_controller = NS(activate=AsyncMock())
        context = NS(user_data={})
        await bot.callbacks.handle(update(data='ui:activate:42'), context)
        bot.runtime_controller.activate.assert_not_awaited()
        context.user_data['pending_dialog'] = 42
        await bot.callbacks.handle(update(data='ui:activate:42', owner=2), context)
        bot.runtime_controller.activate.assert_not_awaited()
        await bot.callbacks.handle(update(data='ui:activate:42'), context)
        bot.runtime_controller.activate.assert_awaited_once()
        self.assertNotIn('pending_dialog', context.user_data)

    async def test_mood_text_does_not_generate(self):
        bot = self.bot()
        bot.send_generated_answers = AsyncMock()
        context = NS(user_data={})
        await bot.callbacks.handle(update(data='ui:set_mood'), context)
        await bot.text_instruction(update(), context)
        self.assertEqual(bot.state.current_mood, 'спокойное')
        bot.send_generated_answers.assert_not_awaited()

    async def test_switch_commits_workspace_together_and_drains_listener(self):
        bot = self.bot()
        dialog = NS(id=42, name='target')
        bot.agent_manager.find_dialog = AsyncMock(return_value=dialog)
        controller = RuntimeController(Mock(), NS(id=1), bot)
        bot.runtime_controller = controller
        old = NS(unregister=AsyncMock())
        controller.listener = old
        workspace = NS(chat_history='unused', episodes='unused', episode_embeddings_live='unused')
        runtime = NS(workspace=workspace, dialog_name='target', recent_messages=[],
                     raw_history=[], known_message_ids=set(), reply_service=object(), episode_tracker=object())
        entered, release = asyncio.Event(), asyncio.Event()
        async def prepare(*args, **kwargs):
            old.unregister.assert_awaited_once()
            entered.set()
            await release.wait()
            return runtime
        listener = NS(register=Mock(), unregister=AsyncMock(), message_processing_lock=asyncio.Lock())
        with patch('mybot.services.runtime_controller.prepare_dialog_runtime', side_effect=prepare), \
             patch('mybot.services.runtime_controller.LiveEvents', return_value=listener), \
             patch('mybot.services.runtime_controller.get_last_saved_message_id', return_value=10), \
             patch('mybot.services.runtime_controller.fetch_messages_after', new_callable=AsyncMock, return_value=[]):
            await controller.activate(42, AsyncMock())
            await entered.wait()
            self.assertTrue(controller.busy)
            self.assertIsNone(bot.workspace)
            await bot.send_generated_answers(update())
            release.set()
            await controller.task
        self.assertIs(bot.workspace, workspace)
        self.assertIs(bot.reply_service, runtime.reply_service)
        self.assertIs(bot.recent_messages, runtime.recent_messages)
        self.assertFalse(controller.busy)
        listener.register.assert_called_once()
        await controller.stop()
        listener.unregister.assert_awaited_once()

    async def test_failed_preparation_disables_stale_runtime(self):
        bot = self.bot()
        bot.agent_manager.find_dialog = AsyncMock(return_value=NS(id=42))
        bot.workspace = object()
        bot.reply_service = object()
        controller = RuntimeController(Mock(), NS(id=1), bot)
        controller.listener = NS(unregister=AsyncMock())
        with patch('mybot.services.runtime_controller.prepare_dialog_runtime', new_callable=AsyncMock,
                   side_effect=RuntimeError('tracker mismatch')):
            await controller.activate(42, AsyncMock())
            await controller.task
        self.assertIsNone(bot.workspace)
        self.assertIsNone(bot.reply_service)
        self.assertIsNone(controller.listener)
        self.assertFalse(controller.busy)

    async def test_unregistered_events_cannot_write(self):
        listener = LiveEvents(Mock(), NS(id=42), NS(id=1), 'dialog', NS(), [], [], set(), Mock(), Mock())
        listener.register()
        await listener.unregister()
        with patch('mybot.telegram.live_events.message_to_data', new_callable=AsyncMock) as convert:
            await listener.on_new_message(NS())
            convert.assert_not_awaited()
        listener.client.remove_event_handler.assert_any_call(listener.on_new_message)
        listener.client.remove_event_handler.assert_any_call(listener.on_message_deleted)

    def test_pagination_stable_signed_ids(self):
        dialogs = [NS(id=-1000000-i, name=str(i)) for i in range(19)]
        keyboard, page, pages = dialogs_menu(dialogs, 1)
        self.assertEqual((page, pages), (2, 3))
        self.assertEqual(keyboard.inline_keyboard[0][0].callback_data, 'ui:select:-1000008:1')
        self.assertEqual(dialogs_menu([], 999)[1:], (1, 1))
        for row in keyboard.inline_keyboard:
            for button in row:
                self.assertLessEqual(len(button.callback_data.encode()), 64)


if __name__ == '__main__':
    unittest.main()

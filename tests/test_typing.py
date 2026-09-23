"""Typing indicators: fake Telegram transport, no messages or network traffic."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from telethon import types
from mybot.services.typing_service import TypingService
from mybot.services.runtime_controller import RuntimeController
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.live_events import LiveEvents
from mybot.app.session_state import SessionState


class TypingTests(unittest.IsolatedAsyncioTestCase):
    def service(self, **kwargs):
        client = AsyncMock()
        client.get_input_entity.side_effect = lambda dialog_id: f'peer:{dialog_id}'
        kwargs.setdefault('think_delay', 0)
        service = TypingService(client, **kwargs)
        self.addAsyncCleanup(service.stop)
        return service, client

    async def test_renews_same_peer_and_stops_without_sending_text(self):
        service, client = self.service(interval=.005)
        await service.start(42)
        client.send_read_acknowledge.assert_awaited_once_with('peer:42')
        first = service.task
        await service.start(42)
        self.assertIs(first, service.task)
        await asyncio.sleep(.02)
        self.assertGreaterEqual(client.await_count, 2)
        await service.stop()
        calls = [call.args[0] for call in client.await_args_list]
        self.assertTrue(all(request.peer == 'peer:42' for request in calls))
        self.assertTrue(all(isinstance(request.action, (types.SendMessageTypingAction,
                             types.SendMessageCancelAction)) for request in calls))
        self.assertIsInstance(calls[-1].action, types.SendMessageCancelAction)
        self.assertIsNone(service.dialog_id)
        count = client.await_count
        await asyncio.sleep(.01)
        self.assertEqual(client.await_count, count)
        client.send_message.assert_not_called()

    async def test_immediate_stop_and_wrong_dialog_stop(self):
        service, client = self.service()
        await service.start(-42)
        await service.stop(13)
        self.assertEqual(service.dialog_id, -42)
        await service.stop(-42)
        self.assertIsInstance(client.await_args.args[0].action, types.SendMessageCancelAction)
        self.assertIsNone(service.dialog_id)

    async def test_auto_expiry(self):
        service, client = self.service(duration=.01, interval=.005)
        await service.start(42)
        await asyncio.wait_for(service.task, 1)
        self.assertIsNone(service.dialog_id)
        self.assertIsInstance(client.await_args.args[0].action, types.SendMessageCancelAction)

    async def test_failure_stops_loop_and_can_restart(self):
        service, client = self.service(interval=.005)
        await service.start(42)
        client.side_effect = RuntimeError('offline')
        await asyncio.wait_for(service.task, 1)
        self.assertIsNone(service.dialog_id)
        self.assertIsNotNone(service.last_error)
        client.side_effect = None
        await service.start(13)
        self.assertEqual(service.dialog_id, 13)
        self.assertIsNone(service.last_error)

    async def test_first_request_failure_leaves_no_task(self):
        service, client = self.service()
        client.send_read_acknowledge.side_effect = RuntimeError('offline')
        with self.assertRaises(RuntimeError):
            await service.start(42)
        self.assertIsNone(service.dialog_id)
        self.assertIsNone(service.task)
        client.assert_not_awaited()

    async def test_read_then_think_then_type_and_cancel_during_pause(self):
        service, client = self.service(think_delay=.05, interval=.005)
        await service.start(42)
        client.send_read_acknowledge.assert_awaited_once_with('peer:42')
        self.assertEqual(service.phase, 'thinking')
        self.assertEqual(client.await_count, 0)
        await asyncio.sleep(.01)
        self.assertEqual(client.await_count, 0)
        await service.stop()
        self.assertIsNone(service.phase)
        self.assertTrue(all(not isinstance(call.args[0].action, types.SendMessageTypingAction)
                            for call in client.await_args_list))

        client.reset_mock()
        await service.start(42)
        self.assertEqual(service.phase, 'thinking')
        await asyncio.sleep(.07)
        self.assertEqual(service.phase, 'typing')
        self.assertTrue(any(isinstance(call.args[0].action, types.SendMessageTypingAction)
                            for call in client.await_args_list))

    async def test_unconfirmed_read_never_types(self):
        service, client = self.service()
        client.send_read_acknowledge.return_value = False
        with self.assertRaises(RuntimeError):
            await service.start(42)
        self.assertIsNone(service.phase)
        client.assert_not_awaited()

    async def test_request_timeout_does_not_keep_renewing(self):
        service, client = self.service(interval=.001, request_timeout=.01)
        await service.start(42)
        async def never_returns(*args, **kwargs):
            await asyncio.Event().wait()
        client.side_effect = never_returns
        await asyncio.wait_for(service.task, 1)
        self.assertIsNone(service.dialog_id)
        self.assertIsNotNone(service.last_error)

    async def test_switch_cancels_old_peer_before_new(self):
        service, client = self.service()
        await service.start(42)
        await service.start(13)
        await asyncio.sleep(.01)
        calls = [call.args[0] for call in client.await_args_list]
        self.assertTrue(any(request.peer == 'peer:42' and
                            isinstance(request.action, types.SendMessageCancelAction)
                            for request in calls))
        self.assertTrue(all(request.peer == 'peer:13' for request in calls
                            if isinstance(request.action, types.SendMessageTypingAction)))
        self.assertEqual(client.send_read_acknowledge.await_count, 2)

    async def test_ui_owner_active_dialog_and_stale_button_guards(self):
        service, _ = self.service()
        bot = BotInterface(1, [], None, SessionState(), telegram_client=Mock())
        bot.menu = NS(show=AsyncMock())
        bot.workspace = NS(dialog_id=42, dialog_name='test')
        bot.runtime_controller = NS(typing=service, busy=False, lock=asyncio.Lock())
        update = NS(effective_message=Mock(), effective_user=NS(id=2),
                    effective_chat=NS(id=2, type='private'))
        context = NS(user_data={})
        await bot.callbacks.dispatch('ui:typing_start:42', update, context)
        self.assertIsNone(service.task)
        update.effective_user.id = update.effective_chat.id = 1
        await bot.callbacks.dispatch('ui:typing_start:13', update, context)
        self.assertIsNone(service.task)
        bot.runtime_controller.busy = True
        await bot.callbacks.dispatch('ui:typing_start:42', update, context)
        self.assertIsNone(service.task)
        bot.runtime_controller.busy = False
        await bot.callbacks.dispatch('ui:typing_start:42', update, context)
        self.assertEqual(service.dialog_id, 42)
        await bot.callbacks.dispatch('ui:typing_stop', update, context)
        self.assertIsNone(service.dialog_id)
        bot.workspace = None
        await bot.callbacks.dispatch('ui:typing_start:42', update, context)
        self.assertIsNone(service.dialog_id)

    async def test_runtime_switch_failure_and_shutdown_stop_indicator(self):
        service, client = self.service()
        bot = NS(agent_manager=NS(find_dialog=AsyncMock(return_value=None)), control_bot_id=999)
        with tempfile.TemporaryDirectory() as root, patch(
                'mybot.services.runtime_controller.get_global_root', return_value=Path(root)):
            controller = RuntimeController(client, NS(id=1), bot)
            controller.typing = service
            await service.start(42)
            await controller.activate(13, AsyncMock())
            await controller.task
            self.assertIsNone(service.dialog_id)
            await service.start(42)
            await controller.stop()
            self.assertIsNone(service.dialog_id)

    async def test_pause_before_preparation_starts_stops_typing(self):
        service, client = self.service()
        bot = NS()
        with tempfile.TemporaryDirectory() as root, patch(
                'mybot.services.runtime_controller.get_global_root', return_value=Path(root)):
            controller = RuntimeController(client, NS(id=1), bot)
            controller.typing = service
            await service.start(42)
            await controller.activate(13, AsyncMock())
            await controller.pause()
            self.assertIsNone(service.dialog_id)
            self.assertFalse(controller.busy)

    async def test_real_outgoing_stops_only_this_dialog(self):
        service, _ = self.service()
        await service.start(42)
        listener = LiveEvents(Mock(), NS(id=42), NS(id=1), 'test',
                              NS(chat_history='unused', episodes='unused', episode_embeddings_live='unused'),
                              [], [], set(), NS(send_incoming_message=AsyncMock(),
                                                 send_outgoing_message=AsyncMock()), Mock(),
                              typing_service=service)
        listener.enabled = True
        with patch('mybot.telegram.live_events.message_to_data', new_callable=AsyncMock,
                   return_value={'message_id': 1, 'sender': 'Я', 'text': 'hello'}), \
             patch('mybot.telegram.live_events.record_outgoing_reply'), \
             patch('mybot.telegram.live_events.append_message_data'), \
             patch('mybot.telegram.live_events.process_live_episode_message', new_callable=AsyncMock,
                   return_value=None), patch('builtins.print'):
            await listener.on_new_message(NS(message=Mock()))
        self.assertIsNone(service.dialog_id)
        self.assertEqual(listener.recent_messages[0]['text'], 'hello')
        listener.bot_interface.send_incoming_message.assert_not_awaited()

    async def test_new_incoming_reads_again_and_restarts_pause(self):
        service, client = self.service(think_delay=.05, interval=.005)
        await service.start(42)
        await asyncio.sleep(.06)
        self.assertEqual(service.phase, 'typing')
        listener = LiveEvents(Mock(), NS(id=42), NS(id=1), 'test',
                              NS(chat_history='unused', episodes='unused', episode_embeddings_live='unused'),
                              [], [], set(), NS(send_incoming_message=AsyncMock()), Mock(),
                              typing_service=service)
        listener.enabled = True
        with patch('mybot.telegram.live_events.message_to_data', new_callable=AsyncMock,
                   return_value={'message_id': 2, 'sender': 'Собеседник', 'text': 'new'}), \
             patch('mybot.telegram.live_events.append_message_data'), \
             patch('mybot.telegram.live_events.process_live_episode_message', new_callable=AsyncMock,
                   return_value=None), patch('builtins.print'):
            await listener.on_new_message(NS(message=Mock()))
        self.assertEqual(client.send_read_acknowledge.await_count, 2)
        self.assertEqual(service.phase, 'thinking')
        requests = [call.args[0] for call in client.await_args_list]
        self.assertIsInstance(requests[-1].action, types.SendMessageCancelAction)
        await asyncio.sleep(.06)
        self.assertEqual(service.phase, 'typing')

    async def test_incoming_after_stop_does_not_restart_typing(self):
        service, client = self.service()
        await service.start(42)
        await service.stop()
        await service.on_incoming(42)
        self.assertIsNone(service.dialog_id)
        self.assertEqual(client.send_read_acknowledge.await_count, 1)

    async def test_failed_reread_stops_old_typing(self):
        service, client = self.service()
        await service.start(42)
        client.send_read_acknowledge.side_effect = RuntimeError('offline')
        with self.assertRaises(RuntimeError):
            await service.on_incoming(42)
        self.assertIsNone(service.dialog_id)
        self.assertIsNotNone(service.last_error)

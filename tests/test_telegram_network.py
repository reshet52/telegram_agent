import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
import asyncio
import tempfile
from pathlib import Path

import httpx
from telegram.error import TimedOut, BadRequest

from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.account_bot import AccountBot


class TelegramNetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_reconnects_with_bounded_backoff(self):
        from mybot.telegram.startup_network import startup_request
        call = AsyncMock(side_effect=[TimedOut()] * 6 + ['connected'])
        with patch('mybot.telegram.startup_network.asyncio.sleep', AsyncMock()) as sleep:
            self.assertEqual(await startup_request(call), 'connected')
        self.assertEqual([c.args[0] for c in sleep.await_args_list], [2, 4, 8, 16, 30, 30])

    async def test_startup_does_not_retry_token_or_request_errors(self):
        from mybot.telegram.startup_network import startup_request
        from telegram.error import InvalidToken
        for error in (InvalidToken('invalid'), BadRequest('invalid')):
            call = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await startup_request(call)
            call.assert_awaited_once()

    async def test_startup_wait_can_be_cancelled(self):
        from mybot.telegram.startup_network import startup_request
        call = AsyncMock(side_effect=TimedOut())
        with patch('mybot.telegram.startup_network.asyncio.sleep', AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await startup_request(call)
        call.assert_awaited_once()

    async def test_lesson_opens_in_clicked_menu_not_old_card(self):
        from mybot.telegram.bot_style_lesson import BotStyleLesson
        from mybot.storage.style_lesson import StyleLesson
        from mybot.storage.workspace import Workspace
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temp:
            ws = Workspace(1, 2, 'Test', Path(temp))
            lesson = StyleLesson(ws)
            lesson.write('style_card.json', {'message_id': 10})
            bot = NS(is_owner=lambda _: True, goal=NS(workspace=lambda: ws),
                     runtime_controller=NS(lock=asyncio.Lock()), menu=NS(message=object(), _remember=Mock()))
            ui = BotStyleLesson(bot)
            ui.overview = AsyncMock()
            await ui.handle('ui:lesson', NS(callback_query=object(), effective_message=NS(message_id=99)), None)
            self.assertEqual(lesson.read('style_card.json')['message_id'], 99)
            self.assertIsNone(bot.menu.message)
            ui.overview.assert_awaited_once_with(ws)

    async def test_card_edit_retries_but_ambiguous_send_does_not(self):
        from mybot.telegram.bot_style_lesson import screen_request
        edit = AsyncMock(side_effect=[TimedOut(), None])
        send = AsyncMock(side_effect=TimedOut())
        with patch('mybot.telegram.bot_style_lesson.asyncio.sleep', AsyncMock()):
            await screen_request(edit, editing=True, text='card')
            with self.assertRaises(TimedOut):
                await screen_request(send, text='card')
        self.assertEqual(edit.await_count, 2)
        send.assert_awaited_once()

    def menu(self, error):
        api = NS(send_message=AsyncMock(side_effect=[error, NS(message_id=12)]))
        return BotMenu(NS(owner_id=1, application=NS(bot=api))), api

    async def test_connect_timeout_retries_menu_once(self):
        error = TimedOut()
        error.__cause__ = httpx.ConnectTimeout('connection')
        menu, api = self.menu(error)
        with patch('mybot.telegram.bot_menu.asyncio.sleep', AsyncMock()):
            await menu.show(None, 'menu')
        self.assertEqual(api.send_message.await_count, 2)
        self.assertEqual(menu.message.message_id, 12)

    async def test_ambiguous_timeout_does_not_duplicate_send(self):
        for cause in (None, httpx.ReadTimeout('read'), httpx.WriteTimeout('write')):
            error = TimedOut()
            error.__cause__ = cause
            menu, api = self.menu(error)
            with self.assertRaises(TimedOut):
                await menu.show(None, 'menu')
            api.send_message.assert_awaited_once()

    async def test_retries_are_bounded(self):
        error = TimedOut()
        error.__cause__ = httpx.ConnectTimeout('connection')
        menu, api = self.menu(error)
        api.send_message.side_effect = error
        with patch('mybot.telegram.bot_menu.asyncio.sleep', AsyncMock()), self.assertRaises(TimedOut):
            await menu.show(None, 'menu')
        self.assertEqual(api.send_message.await_count, 2)

    async def test_network_logged_briefly_other_errors_keep_traceback(self):
        service = AccountBot.__new__(AccountBot)
        with patch('mybot.telegram.account_bot.logging.warning') as warning, \
             patch('mybot.telegram.account_bot.logging.error') as log:
            await service._error(None, NS(error=TimedOut()))
            warning.assert_called_once()
            log.assert_not_called()
            error = BadRequest('invalid markup')
            await service._error(None, NS(error=error))
            self.assertIs(log.call_args.kwargs['exc_info'][1], error)

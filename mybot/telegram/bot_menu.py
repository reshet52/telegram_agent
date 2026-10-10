"""A single transient menu message; generated content remains in the chat."""

import asyncio
import logging
import httpx

from telegram.error import BadRequest, TelegramError, TimedOut
from mybot.telegram.bot_panel import PANEL_TEXT, PANEL_HIDDEN_TEXT


class BotMenu:
    def __init__(self, interface):
        self.interface = interface
        self.message = None
        self.lock = asyncio.Lock()

    def _saved_id(self):
        state = getattr(self.interface, "control_state", None)
        return state.get("menu_message_id") if state else None

    def _remember(self, message_id):
        state = getattr(self.interface, "control_state", None)
        if state:
            state.set(menu_message_id=message_id)

    async def move_to_bottom(self):
        async with self.lock:
            message_id = self.message.message_id if self.message is not None else self._saved_id()
            if message_id is not None:
                try:
                    await self.interface.application.bot.delete_message(
                        chat_id=self.interface.owner_id, message_id=message_id)
                except TelegramError:
                    pass
                self.message = None
                self._remember(None)

    async def show(self, source, text, reply_markup=None):
        async with self.lock:
            try:
                await self._show(source, text, reply_markup)
            except TimedOut as error:
                # A connect timeout precedes delivery. Read/write timeouts can mean
                # Telegram accepted the message: do not duplicate those sends.
                if not isinstance(error.__cause__, httpx.ConnectTimeout):
                    raise
                logging.warning('Не удалось подключиться к Telegram для меню; повтор через секунду.')
                await asyncio.sleep(1)
                await self._show(source, text, reply_markup)

    async def _show(self, source, text, reply_markup=None):
        if source is not None and self.is_variant_message(source):
            source = None
        # A callback may belong to a menu left by the previous process.
        if source is not None and getattr(source, "reply_markup", None):
            saved_id = self._saved_id()
            if saved_id and source.message_id != saved_id:
                try:
                    await source.delete()
                except BadRequest:
                    pass
            elif self.message is None:
                self.message = source
            elif source.message_id != self.message.message_id:
                try:
                    await source.delete()
                except BadRequest:
                    pass
        if self.message is not None:
            try:
                await self.message.edit_text(text, reply_markup=reply_markup)
                self._remember(self.message.message_id)
                return
            except BadRequest as error:
                if "message is not modified" in str(error).lower():
                    return
                if not any(value in str(error).lower() for value in (
                    "message to edit not found", "message can't be edited",
                )):
                    raise
        elif self._saved_id():
            try:
                await self.interface.application.bot.edit_message_text(
                    chat_id=self.interface.owner_id, message_id=self._saved_id(),
                    text=text, reply_markup=reply_markup)
                return
            except BadRequest as error:
                if "message is not modified" in str(error).lower():
                    return
                if not any(value in str(error).lower() for value in (
                    "message to edit not found", "message can't be edited")):
                    raise
        self.message = await self.interface.application.bot.send_message(
            chat_id=self.interface.owner_id, text=text, reply_markup=reply_markup,
        )
        self._remember(self.message.message_id)

    @staticmethod
    def is_variant_message(message):
        markup = getattr(message, "reply_markup", None)
        if markup is None:
            return False
        # python-telegram-bot objects and Telethon history expose different rows.
        rows = getattr(markup, "inline_keyboard", None)
        if rows is None:
            rows = [row.buttons for row in getattr(markup, "rows", [])]
        for row in rows:
            for button in row:
                data = getattr(button, "callback_data", None) or getattr(button, "data", None)
                if isinstance(data, bytes) and data.startswith((b'ui:sl:', b'ui:s2:', b'ui:learn:')):
                    return True
                if isinstance(data, str) and data.startswith(('ui:sl:', 'ui:s2:', 'ui:learn:')):
                    return True
                if isinstance(data, bytes) and data.startswith((
                        b"ui:rv:", b"ui:rve:", b"ui:draft:", b"ui:db:",
                        b"ui:unreject:", b"ui:fb:", b"ui:rr:", b"ui:rp:rate:",
                        b"ui:rp:run:", b"ui:rp:reveal:", b"ui:rp:view:", b"ui:bm:add:", b"ui:bm:rate_")):
                    return True
                if isinstance(data, str) and data.startswith((
                        "ui:rv:", "ui:rve:", "ui:draft:", "ui:db:",
                        "ui:unreject:", "ui:fb:", "ui:rr:", "ui:rp:rate:",
                        "ui:rp:run:", "ui:rp:reveal:", "ui:rp:view:", "ui:bm:add:", "ui:bm:rate_")):
                    return True
        return False


    async def clean_previous_menus(self):
        """Remove only recognizable bot-owned UI messages in the recent chat."""
        bot = self.interface.application.bot
        client = self.interface.telegram_client
        if client is None:
            return
        try:
            async for item in client.iter_messages(bot.username, limit=200):
                if item.sender_id != bot.id:
                    continue
                if item.id == self._saved_id():
                    continue
                if getattr(item, 'text', None) in {PANEL_TEXT, PANEL_HIDDEN_TEXT}:
                    try:
                        await bot.delete_message(chat_id=self.interface.owner_id, message_id=item.id)
                    except TelegramError:
                        pass
                    continue
                if not item.reply_markup:
                    continue
                buttons = [button for row in getattr(item.reply_markup, "rows", [])
                           for button in row.buttons]
                if not buttons or not all(
                    isinstance(getattr(button, "data", None), bytes)
                    and button.data.startswith(b"ui:") for button in buttons
                ):
                    continue
                if self.is_variant_message(item):
                    # Generated variants are content, not transient navigation.
                    continue
                try:
                    await bot.delete_message(chat_id=self.interface.owner_id, message_id=item.id)
                except TelegramError:
                    # Telegram may refuse deletion of older messages.
                    continue
        except Exception:
            logging.warning("Не удалось очистить прежние меню; работа бота продолжается.")

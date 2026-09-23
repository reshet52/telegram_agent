"""A single transient menu message; generated content remains in the chat."""

import asyncio
import logging

from telegram.error import BadRequest, TelegramError
from mybot.telegram.bot_panel import PANEL_TEXT, PANEL_HIDDEN_TEXT


class BotMenu:
    def __init__(self, interface):
        self.interface = interface
        self.message = None
        self.lock = asyncio.Lock()

    async def move_to_bottom(self):
        async with self.lock:
            if self.message is not None:
                try:
                    await self.message.delete()
                except TelegramError:
                    pass
                self.message = None

    async def show(self, source, text, reply_markup=None):
        async with self.lock:
            await self._show(source, text, reply_markup)

    async def _show(self, source, text, reply_markup=None):
        if source is not None and self.is_variant_message(source):
            source = None
        # A callback may belong to a menu left by the previous process.
        if source is not None and getattr(source, "reply_markup", None):
            if self.message is None:
                self.message = source
            elif source.message_id != self.message.message_id:
                try:
                    await source.delete()
                except BadRequest:
                    pass
        if self.message is not None:
            try:
                await self.message.edit_text(text, reply_markup=reply_markup)
                return
            except BadRequest as error:
                if "message is not modified" in str(error).lower():
                    return
                if not any(value in str(error).lower() for value in (
                    "message to edit not found", "message can't be edited",
                )):
                    raise
        self.message = await self.interface.application.bot.send_message(
            chat_id=self.interface.owner_id, text=text, reply_markup=reply_markup,
        )

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
                if isinstance(data, bytes) and data.startswith((
                        b"ui:rv:", b"ui:rve:", b"ui:draft:", b"ui:db:",
                        b"ui:unreject:", b"ui:fb:")):
                    return True
                if isinstance(data, str) and data.startswith((
                        "ui:rv:", "ui:rve:", "ui:draft:", "ui:db:",
                        "ui:unreject:", "ui:fb:")):
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

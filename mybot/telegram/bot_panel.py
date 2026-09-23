"""Persistent keyboard controls next to the Telegram input field."""

from telegram import BotCommand, MenuButtonCommands
from telegram.error import TelegramError
from mybot.telegram.bot_keyboards import control_panel


PANEL_TEXT = 'Пульт управления готов. Выберите действие возле поля ввода.'


class BotPanel:
    def __init__(self, interface):
        self.interface = interface
        self.message = None

    async def install(self):
        api = self.interface.application.bot
        await api.set_my_commands([
            BotCommand('start', 'Пульт управления'),
            BotCommand('dialogs', 'Выбрать диалог'),
            BotCommand('reply', 'Предложить ответ'),
            BotCommand('show', 'Показать контекст'),
            BotCommand('status', 'Прогресс и продолжение подготовки'),
            BotCommand('style', 'Общий стиль общения'),
        ])
        await api.set_chat_menu_button(chat_id=self.interface.owner_id, menu_button=MenuButtonCommands())
        await self.show()

    async def show(self):
        if self.message:
            try:
                await self.message.delete()
            except TelegramError:
                pass
        self.message = await self.interface.application.bot.send_message(
            chat_id=self.interface.owner_id, text=PANEL_TEXT, reply_markup=control_panel())

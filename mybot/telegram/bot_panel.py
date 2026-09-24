"""Persistent keyboard controls next to the Telegram input field."""

from telegram import BotCommand, MenuButtonCommands, ReplyKeyboardRemove
from telegram.error import TelegramError
from mybot.telegram.bot_keyboards import control_panel


PANEL_TEXT = 'Пульт управления готов. Выберите действие возле поля ввода.'
PANEL_HIDDEN_TEXT = 'Пульт скрыт. Вернуть: /panel или кнопка в меню.'


class BotPanel:
    def __init__(self, interface):
        self.interface = interface
        self.message = None
        self.hidden = bool(interface.control_state.get("panel_hidden", False)) if interface.control_state else False

    async def install(self):
        api = self.interface.application.bot
        await api.set_my_commands([
            BotCommand('start', 'Пульт управления'),
            BotCommand('dialogs', 'Выбрать диалог'),
            BotCommand('reply', 'Предложить ответ'),
            BotCommand('show', 'Показать контекст'),
            BotCommand('status', 'Прогресс и продолжение подготовки'),
            BotCommand('style', 'Общий стиль общения'),
            BotCommand('panel', 'Показать пульт'),
        ])
        await api.set_chat_menu_button(chat_id=self.interface.owner_id, menu_button=MenuButtonCommands())
        if self.hidden:
            await self.hide()
        else:
            await self.show()

    async def _replace(self, text, markup):
        state = self.interface.control_state
        old_id = self.message.message_id if self.message else (state.get("panel_message_id") if state else None)
        if old_id:
            try:
                if self.message is not None:
                    await self.message.delete()
                else:
                    await self.interface.application.bot.delete_message(
                        chat_id=self.interface.owner_id, message_id=old_id)
            except TelegramError:
                pass
        self.message = await self.interface.application.bot.send_message(
            chat_id=self.interface.owner_id, text=text, reply_markup=markup)
        if state:
            state.set(panel_message_id=self.message.message_id, panel_hidden=self.hidden)

    async def show(self):
        self.hidden = False
        await self._replace(PANEL_TEXT, control_panel(active=self.interface.workspace is not None))

    async def hide(self):
        self.hidden = True
        await self._replace(PANEL_HIDDEN_TEXT, ReplyKeyboardRemove())

    async def refresh(self):
        if not self.hidden:
            await self.show()

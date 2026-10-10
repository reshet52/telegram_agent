"""Inline navigation and explicit dialog activation."""

import logging
import asyncio

from telegram import InlineKeyboardButton
from telegram.error import TelegramError, TimedOut
from mybot.telegram.bot_keyboards import dialogs_menu, home_menu
from mybot.telegram.bot_benchmark import clear_benchmark_input


class BotCallbacks:
    def __init__(self, interface):
        self.interface = interface

    async def show(self, message, text, reply_markup=None):
        controller = self.interface.runtime_controller
        if reply_markup is None and controller and controller.busy and controller.can_cancel:
            reply_markup = home_menu([[
                InlineKeyboardButton("⏹ Отменить обновление", callback_data="ui:cancel_update")]])
        await self.interface.menu.show(message, text, reply_markup or home_menu())

    async def handle(self, update, context):
        query = update.callback_query
        if not self.interface.is_owner(update):
            await query.answer()
            return
        action = query.data or ""
        try:
            await asyncio.wait_for(query.answer(), timeout=2)
        except (TimedOut, asyncio.TimeoutError):
            # Acknowledgement only dismisses Telegram's loading indicator.
            # Do not retry the action or discard it because this request timed out.
            logging.warning('Telegram не подтвердил нажатие вовремя; продолжаю обработку команды.')
        except TelegramError:
            if query.data not in {"ui:pause", "ui:cancel_update"}:
                raise
        await self.dispatch(action, update, context)

    async def dispatch(self, action, update, context):
        if not self.interface.is_owner(update):
            return
        message = update.effective_message
        if message is None:
            return
        bot = self.interface
        from mybot.telegram.bot_mixed_replies import cancel_clarification
        cancel_clarification(bot)
        if action in {'ui:home', 'ui:learn', 'ui:reply_instruction'} or action.startswith(('ui:dialogs:', 'ui:select:', 'ui:activate:')):
            task = getattr(bot, 'mixed_task', None)
            if task and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        allowed = {'ui:home', 'ui:reply', 'ui:reply_force', 'ui:reply_instruction', 'ui:learn', 'ui:cancel_update'}
        prefixes = ('ui:dialogs:', 'ui:select:', 'ui:activate:', 'ui:rv:', 'ui:rve:', 'ui:draft:',
                    'ui:opts:', 'ui:db:', 'ui:mr:', 'ui:reject:', 'ui:unreject:', 'ui:learn:')
        if not action.startswith('ui:learn'):
            bot.simple_lesson.leave()
            if bot.simple_lesson.task and not bot.simple_lesson.task.done():
                await asyncio.gather(bot.simple_lesson.task, return_exceptions=True)
        if action not in allowed and not action.startswith(prefixes):
            for field in ('awaiting_reply_instruction', 'awaiting_history_search', 'awaiting_goal',
                          'awaiting_behavior', 'practice_correction', 'pending_reply_edit'):
                context.user_data.pop(field, None)
            await self.show(message, 'Меню изменилось.', bot.main_menu())
            return
        clear_benchmark_input(context)
        for field in ('awaiting_mood', 'awaiting_goal', 'awaiting_behavior', 'awaiting_history_search', 'practice_correction'):
            context.user_data.pop(field, None)
        if action != 'ui:reply_instruction':
            context.user_data.pop('awaiting_reply_instruction', None)
        if not action.startswith('ui:draft:edit:'):
            context.user_data.pop('pending_reply_edit', None)
        if not action.startswith('ui:activate:'):
            context.user_data.pop('pending_dialog', None)
        if action == 'ui:learn' or action.startswith('ui:learn:'):
            await bot.simple_lesson.handle(action, message)
        elif action == 'ui:home':
            await self.show(message, bot.menu_title(), bot.main_menu())
        elif action == 'ui:cancel_update':
            await bot.runtime_controller.pause()
            await self.show(message, bot.menu_title(), bot.main_menu())
        elif action.startswith('ui:dialogs:'):
            await self.show_dialogs(message, context, action)
        elif action.startswith('ui:select:'):
            await self.select_dialog(message, action, context)
        elif action.startswith('ui:activate:'):
            try:
                dialog_id = int(action.split(':')[2])
            except (ValueError, IndexError):
                return
            if context.user_data.get('pending_dialog') != dialog_id:
                await self.show(message, 'Подтверждение устарело. Выберите диалог заново.')
                return
            context.user_data.pop('pending_dialog', None)
            await bot.runtime_controller.activate(dialog_id, lambda text: self.show(message, text))
        elif action == 'ui:reply':
            await bot.send_generated_answers(update)
        elif action == 'ui:reply_force':
            await bot.send_generated_answers(update, force_reply=True)
        elif action == 'ui:reply_instruction':
            if bot.workspace is None or bot.runtime_controller is None or bot.runtime_controller.busy:
                await self.show(message, 'Сначала откройте диалог и дождитесь подготовки.')
                return
            context.user_data['awaiting_reply_instruction'] = bot.workspace.dialog_id
            await self.show(message, 'Напишите указания для следующего ответа. '
                            'Следующее сообщение будет использовано для генерации. '
                            'Для отмены нажмите «Главное меню».')
        elif action.startswith('ui:mr:'):
            from mybot.telegram.bot_mixed_replies import retry
            await retry(bot, action, message)
        else:
            await bot.reply_actions.handle(action, context, message)

    async def show_dialogs(self, message, context, action):
        try:
            page = int(action.split(":")[2])
            if page == 0 or "ui_dialogs" not in context.user_data:
                dialogs = await self.interface.agent_manager.get_dialogs()
                context.user_data["ui_dialogs"] = [
                    dialog for dialog in dialogs
                    if dialog.id != self.interface.control_bot_id
                ]
            dialogs = context.user_data["ui_dialogs"]
            keyboard, current, pages = dialogs_menu(dialogs, page)
            await self.show(message,
                f"Диалоги — страница {current}/{pages}" if dialogs else "Диалогов нет.", keyboard)
        except Exception as error:
            await self.show(message, f"Не удалось получить диалоги:\n{error}")

    async def select_dialog(self, message, action, context):
        try:
            _, _, dialog_id, page = action.split(":")
            dialog_id, page = int(dialog_id), max(0, int(page))
            controller = self.interface.runtime_controller
            if controller.busy:
                await self.show(message, "Подготовка уже выполняется. Дождитесь завершения.")
                return
            text = await controller.describe(dialog_id)
            context.user_data["pending_dialog"] = dialog_id
            await self.show(message, text, home_menu([
                [InlineKeyboardButton("Подготовить и открыть", callback_data=f"ui:activate:{dialog_id}")],
                [InlineKeyboardButton("⬅️ К диалогам", callback_data=f"ui:dialogs:{page}")],
            ]))
        except Exception as error:
            await self.show(message, f"Не удалось выбрать диалог:\n{error}")

"""Inline navigation and explicit dialog activation."""

from telegram import InlineKeyboardButton
from mybot.services.global_style import load_global_style
from mybot.telegram.bot_keyboards import dialogs_menu, home_menu
from mybot.telegram.bot_typing import handle_typing


class BotCallbacks:
    def __init__(self, interface):
        self.interface = interface

    async def show(self, message, text, reply_markup=None):
        await self.interface.menu.show(message, text, reply_markup or home_menu())

    async def handle(self, update, context):
        query = update.callback_query
        if not self.interface.is_owner(update):
            await query.answer()
            return
        await query.answer()
        await self.dispatch(query.data or "", update, context)

    async def dispatch(self, action, update, context):
        if not self.interface.is_owner(update):
            return
        message = update.effective_message
        if message is None:
            return
        bot = self.interface
        if not action.startswith("ui:activate:"):
            context.user_data.pop("pending_dialog", None)
        if action != "ui:set_mood":
            context.user_data.pop("awaiting_mood", None)
        keep_reply_input = (action.startswith("ui:draft:edit:")
                            or action in {"ui:typing", "ui:typing_stop", "ui:typing_toggle"}
                            or action.startswith("ui:typing_start:"))
        if not keep_reply_input:
            context.user_data.pop("pending_reply_edit", None)
        if action == "ui:home":
            await self.show(message, bot.menu_title(), bot.main_menu())
        elif action == "ui:leave":
            if await bot.runtime_controller.deactivate():
                await self.show(message, "Главное меню. Выберите диалог.", bot.main_menu())
            else:
                await self.show(message, "Дождитесь завершения текущей операции.")
        elif action == "ui:panel_hide":
            await bot.panel.hide()
            await self.show(message, bot.menu_title(), bot.main_menu())
        elif action == "ui:panel_show":
            await bot.panel.show()
            await self.show(message, bot.menu_title(), bot.main_menu())
        elif action in {"ui:typing", "ui:typing_stop", "ui:typing_toggle"} or action.startswith("ui:typing_start:"):
            await handle_typing(bot, action, message)
        elif action.startswith("ui:dialogs:"):
            await self.show_dialogs(message, context, action)
        elif action.startswith("ui:select:"):
            await self.select_dialog(message, action, context)
        elif action.startswith("ui:activate:"):
            try:
                dialog_id = int(action.split(":")[2])
            except (ValueError, IndexError):
                return
            if context.user_data.get("pending_dialog") != dialog_id:
                await self.show(message, "Подтверждение устарело. Выберите диалог заново.")
                return
            context.user_data.pop("pending_dialog", None)
            await bot.runtime_controller.activate(
                dialog_id, lambda text: self.show(message, text))
        elif action == "ui:reply":
            await bot.send_generated_answers(update)
        elif (action.startswith(("ui:rv:", "ui:rve:", "ui:draft:",
                                 "ui:opts:", "ui:db:", "ui:unreject:", "ui:dc:"))
              or action.startswith("ui:reject:")):
            await bot.reply_actions.handle(action, context, message)
        elif action.startswith(("ui:fb:", "ui:fl:", "ui:fi:", "ui:ff:")):
            await bot.feedback.handle(action)
        elif action == "ui:context":
            await bot.show_command(update, context)
        elif action == "ui:mood":
            await self.show(message,
                f"Настроение: {bot.state.current_mood or 'не задано'}\n"
                "Действует только в текущей сессии.",
                home_menu([
                    [InlineKeyboardButton("Задать настроение", callback_data="ui:set_mood")],
                    [InlineKeyboardButton("Сбросить", callback_data="ui:clear_mood")],
                ]))
        elif action == "ui:set_mood":
            context.user_data["awaiting_mood"] = True
            await self.show(message, "Напишите настроение следующим сообщением.\n"
                            "Для отмены нажмите «Главное меню».")
        elif action == "ui:clear_mood":
            bot.state.current_mood = None
            await self.show(message, "Настроение сброшено.")
        elif action == "ui:status":
            controller = bot.runtime_controller
            rows = []
            if controller.busy:
                rows.append([InlineKeyboardButton("⏸ Приостановить", callback_data="ui:pause")])
            elif controller.job.get('status') in {'paused', 'failed'}:
                rows.append([InlineKeyboardButton("▶️ Продолжить", callback_data="ui:resume")])
            await self.show(message, controller.last_status, home_menu(rows))
        elif action == "ui:pause":
            await bot.runtime_controller.pause()
            await self.dispatch("ui:status", update, context)
        elif action == "ui:resume":
            await bot.runtime_controller.resume(lambda text: self.show(message, text))
        elif action == "ui:style":
            style = load_global_style(bot.owner_id)
            labels = {'length': 'Длина', 'formality': 'Формальность', 'emoji': 'Эмодзи',
                      'punctuation': 'Пунктуация', 'message_splitting': 'Разбиение ответа'}
            values = {'short': 'короткие сообщения', 'medium': 'средняя', 'long': 'длинные сообщения',
                      'informal': 'неформальная', 'neutral': 'нейтральная', 'formal': 'формальная',
                      'rare': 'редко', 'moderate': 'умеренно', 'frequent': 'часто',
                      'minimal': 'минимальная', 'standard': 'обычная', 'expressive': 'выразительная',
                      'single': 'одно сообщение', 'multiple': 'несколько сообщений'}
            text = '\n'.join(f'{labels[key]}: {values[value]}' for key, value in style.items())
            await self.show(message, "Общий стиль аккаунта\n\n" + (text or
                "Пока не собран. Запустите полный анализ в разделе «Память».") +
                "\n\nОбщий стиль применяется во всех чатах. Факты о людях сюда не переносятся.")
        elif action == "ui:full_analysis":
            if bot.workspace is None or bot.runtime_controller.busy:
                await self.show(message, "Сначала откройте диалог и дождитесь подготовки.")
                return
            context.user_data['pending_full'] = bot.workspace.dialog_id
            await self.show(message,
                "Проанализировать всю историю активного диалога, собрать память, индексы и общий стиль? "
                "Сохранённые части будут использованы повторно. Новые части выполняют платные запросы API.",
                home_menu([[InlineKeyboardButton("Начать полный анализ", callback_data="ui:confirm_full")]]))
        elif action == "ui:confirm_full":
            dialog_id = context.user_data.pop('pending_full', None)
            if dialog_id is None or bot.workspace is None or dialog_id != bot.workspace.dialog_id:
                await self.show(message, "Подтверждение устарело. Откройте раздел «Память».")
                return
            await bot.runtime_controller.activate(dialog_id, lambda text: self.show(message, text), full_analysis=True)
        elif action == "ui:memory":
            await self.show(message,
                "Обновить память активного диалога? История анализируется частями, "
                "сохранённые анализы используются повторно. Это платные запросы AI. "
                "Для старого диалога без сохранённых этапов потребуется полный анализ.",
                home_menu([[InlineKeyboardButton("Обновить память", callback_data="ui:update_memory")],
                           [InlineKeyboardButton("Полный анализ истории", callback_data="ui:full_analysis")],
                           [InlineKeyboardButton("Прогресс подготовки", callback_data="ui:status")]]))
        elif action == "ui:update_memory":
            await bot.update_memory_command(update, context)
        elif action == "ui:settings":
            await self.show(message, bot.menu_title() + "\n\n"
                "Варианты ответа показываются только вам в этом боте. "
                "Отправка собеседнику отключена; готовый текст можно скопировать.\n"
                "Команды: /dialogs, /select, /reply, /show, /mood, "
                "/clear_mood, /update_memory, /panel")
        else:
            await self.show(message, "Откройте меню заново: /start")

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

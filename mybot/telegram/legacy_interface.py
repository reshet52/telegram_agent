"""Compatibility-only tools. Hidden features load on explicit direct access, not startup."""
import importlib
import logging
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from mybot.telegram.bot_operations import active_operation

class LegacyInterface:
    _legacy_types = {
        'feedback': ('bot_feedback', 'BotFeedback'),
        'benchmark': ('bot_benchmark', 'BotBenchmark'),
        'goal': ('bot_goal', 'BotGoal'),
        'behavior': ('bot_behavior', 'BotBehavior'),
        'reply_practice': ('bot_reply_practice', 'BotReplyPractice'),
        'style_lesson': ('bot_style_lesson', 'BotStyleLesson'),
    }

    def __getattr__(self, name):
        if name == 'random_facts':
            from mybot.services.random_fact import RandomFactService
            value = RandomFactService()
        elif name in self._legacy_types:
            module, classname = self._legacy_types[name]
            factory = getattr(importlib.import_module('mybot.telegram.' + module), classname)
            value = factory(self)
        else:
            raise AttributeError(name)
        setattr(self, name, value)
        return value

    async def panel_command(self, update, context):
        if not self.is_owner(update):
            return
        await self.panel.show()
        await self.menu.show(update.effective_message, self.menu_title(), self.main_menu())


    async def status_command(self, update, context):
        await self.callbacks.dispatch("ui:status", update, context)


    async def style_command(self, update, context):
        await self.callbacks.dispatch("ui:style", update, context)


    @active_operation
    async def show_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        text = self.format_messages(
            self.recent_messages
        )

        if not text:
            text = "Контекст пуст."

        await self.send_long_text(
            update.effective_message,
            text
        )


    async def goal_command(self, update, context):
        if not self.is_owner(update):
            return
        clear_benchmark_input(context)
        for key in ('awaiting_mood', 'awaiting_history_search', 'awaiting_reply_instruction', 'pending_reply_edit'):
            context.user_data.pop(key, None)
        if context.args:
            await self.goal.handle('ui:goal:write', update, context, text=' '.join(context.args))
        else:
            await self.goal.handle('ui:goal', update, context)


    async def clear_goal_command(self, update, context):
        if not self.is_owner(update):
            return
        action = f'ui:goal:clear:{self.workspace.dialog_id}' if self.workspace else 'ui:goal'
        await self.callbacks.dispatch(action, update, context)


    async def search_command(self, update, context):
        if not self.is_owner(update):
            return
        question = " ".join(context.args).strip()
        if question:
            await self.answer_history_question(update, question)
        else:
            await self.callbacks.dispatch("ui:history_search", update, context)


    @active_operation
    async def answer_history_question(self, update, question):
        from mybot.services.history_search import search_history
        await self.callbacks.show(update.effective_message, "Ищу подтверждение во всей сохранённой истории диалога…")
        try:
            answer = await search_history(self.workspace, question)
        except ValueError as error:
            answer = str(error)
        except Exception:
            logging.exception("Не удалось выполнить поиск по истории")
            answer = "Поиск не удался. История не изменена; попробуйте повторить вопрос позже."
        await self.send_long_text(update.effective_message, answer)


    @active_operation
    async def random_fact_command(self, update, expected_dialog=None):
        from mybot.storage.atomic import durable_call
        from mybot.storage.search_benchmark import BenchmarkStore
        from mybot.telegram.bot_benchmark import button as benchmark_button
        workspace = self.workspace
        if expected_dialog is not None and expected_dialog != str(workspace.dialog_id):
            await self.callbacks.show(update.effective_message,
                                      "Откройте исходный диалог или нажмите «Случайный факт» в текущем меню.")
            return
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("🎲 Ещё факт", callback_data=f"ui:random_fact:{workspace.dialog_id}")],
            [InlineKeyboardButton("🏠 Главное меню", callback_data="ui:home")],
        ])
        await self.callbacks.show(update.effective_message, "Ищу случайный факт · эксперимент…")
        try:
            fact = await self.random_facts.extract(workspace)
        except ValueError as error:
            await self.callbacks.show(update.effective_message, str(error), markup)
            return
        except Exception:
            logging.exception("Не удалось извлечь случайный факт")
            await self.callbacks.show(update.effective_message,
                "Не удалось получить факт. Попробуйте позже кнопкой «Ещё факт».", markup)
            return
        if fact is None:
            await self.callbacks.show(update.effective_message,
                "В выбранных фрагментах не нашёл нового подтверждённого факта о собеседнике. "
                "Это не означает, что фактов нет во всей истории.", markup)
            return
        await self.callbacks.show(update.effective_message, "Факт найден · эксперимент", markup)
        try:
            fact_id = await durable_call(BenchmarkStore(workspace).save_fact, fact)
        except Exception:
            logging.exception("Не удалось сохранить факт для проверочного набора")
            await self.callbacks.show(update.effective_message,
                "Факт найден, но не сохранён для проверочного набора. Повторите запрос позже.", markup)
        else:
            markup = InlineKeyboardMarkup([
                [benchmark_button("✅ В проверку", 'add', workspace, fact_id)],
                *markup.inline_keyboard,
            ])
        await self.send_long_text(update.effective_message, fact.format(workspace.dialog_name),
                                  reply_markup=markup)
        self.random_facts.remember(workspace, fact)


    async def update_memory_command(self, update, context):
        if not self.is_owner(update):
            return
        if self.workspace is None or self.runtime_controller is None:
            await self.callbacks.show(update.effective_message,
                                      "Сначала откройте диалог через меню «Диалоги».")
            return
        await self.callbacks.dispatch("ui:update_history", update, context)

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import logging
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

from mybot.app.session_state import (
    SessionState
)
from mybot.config import Config
from mybot.storage.reply_journal import record_generation
from mybot.services.reply_service import (
    ReplyService
)
from mybot.services.agent_manager import (
    AgentManager
)

from mybot.telegram.bot_keyboards import main_menu, PANEL_ACTIONS
from mybot.telegram.bot_panel import BotPanel
from mybot.telegram.bot_callbacks import BotCallbacks
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_operations import active_operation
from mybot.services.history_search import search_history
from mybot.services.random_fact import RandomFactService
from mybot.telegram.bot_reply_actions import BotReplyActions, variants_markup, variants_text
from mybot.telegram.bot_feedback import BotFeedback
from mybot.telegram.bot_benchmark import BotBenchmark, clear_benchmark_input, button as benchmark_button
from mybot.storage.search_benchmark import BenchmarkStore
from mybot.storage.atomic import durable_call
from mybot.telegram.bot_goal import BotGoal
from mybot.telegram.bot_behavior import BotBehavior
from mybot.ai.reply_provider import reply_model
from mybot.telegram.bot_reply_review import review_markup
from mybot.telegram.bot_reply_practice import BotReplyPractice


class BotInterface:
    def __init__(
        self,
        owner_id,
        recent_messages,
        reply_service: ReplyService,
        state: SessionState,
        workspace=None,
        telegram_client=None
    ):
        self.owner_id = owner_id
        self.recent_messages = (
            recent_messages
        )
        self.reply_service = (
            reply_service
        )
        self.state = state
        self.workspace = workspace

        self.telegram_client = (
            telegram_client
        )
        self.agent_manager = (
            AgentManager(
                telegram_client
            )
            if telegram_client
            else None
        )

        self.application = None
        self.control_state = None
        self.callbacks = BotCallbacks(self)
        self.reply_actions = BotReplyActions(self)
        self.feedback = BotFeedback(self)
        self.menu = BotMenu(self)
        self.panel = BotPanel(self)
        self.runtime_controller = None
        self.control_bot_id = None
        self.random_facts = RandomFactService()
        self.benchmark = BotBenchmark(self)
        self.goal = BotGoal(self)
        self.behavior = BotBehavior(self)
        self.reply_practice = BotReplyPractice(self)


    def is_owner(
        self,
        update
    ):
        user = update.effective_user
        chat = update.effective_chat

        if not user or not chat:
            return False

        return (
            user.id == self.owner_id
            and chat.id == self.owner_id
            and chat.type == "private"
        )


    def format_messages(
        self,
        messages
    ):
        lines = []

        for message in messages:
            sender = message.get(
                "sender",
                "?"
            )

            text = message.get(
                "text"
            )

            if not text:
                message_type = (
                    message.get(
                        "type",
                        "unknown"
                    )
                )

                text = (
                    f"[{message_type}]"
                )

            lines.append(
                f"{sender}: {text}"
            )

        return "\n\n".join(
            lines
        )


    async def send_long_text(
        self,
        message,
        text,
        chunk_size=4096,
        reply_markup=None,
    ):
        for start in range(
            0,
            len(text),
            chunk_size
        ):
            await message.reply_text(
                text[
                    start:
                    start + chunk_size
                ],
                reply_markup=reply_markup if start + chunk_size >= len(text) else None,
            )


    def menu_title(self):
        active = self.workspace.dialog_name if self.workspace else "не выбран"
        return f"AI Agent\nАктивный диалог: {active}"

    async def start_command(self, update, context):
        if not self.is_owner(update):
            return
        clear_benchmark_input(context)
        context.user_data.pop("awaiting_mood", None)
        context.user_data.pop("awaiting_goal", None)
        context.user_data.pop("awaiting_behavior", None)
        context.user_data.pop('practice_correction', None)
        context.user_data.pop("pending_reply_edit", None)
        await self.panel.show()
        await self.menu.move_to_bottom()
        await self.menu.show(update.effective_message, self.menu_title(),
                             self.main_menu())

    def main_menu(self):
        return main_menu(active=self.workspace is not None,
                         panel_hidden=self.panel.hidden)

    async def panel_command(self, update, context):
        if not self.is_owner(update):
            return
        await self.panel.show()
        await self.menu.show(update.effective_message, self.menu_title(), self.main_menu())

    async def status_command(self, update, context):
        await self.callbacks.dispatch("ui:status", update, context)

    async def style_command(self, update, context):
        await self.callbacks.dispatch("ui:style", update, context)

    async def select_dialog_command(self, update, context):
        if not self.is_owner(update):
            return
        try:
            dialog_id = int(context.args[0])
        except (ValueError, IndexError):
            await self.callbacks.show(update.effective_message, "Укажите /select <ID диалога>.")
            return
        await self.callbacks.select_dialog(
            update.effective_message, f"ui:select:{dialog_id}:0", context)

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

    async def dialogs_command(self, update, context):
        if not self.is_owner(update):
            return
        await self.callbacks.show_dialogs(update.effective_message, context, "ui:dialogs:0")


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


    @active_operation
    async def send_generated_answers(
        self,
        update,
        instruction=None
    ):
        if not self.is_owner(
            update
        ):
            return

        await self.callbacks.show(update.effective_message, "Генерирую ответ…")

        try:
            answers = (
                await self.reply_service.generate(
                    instruction=instruction
                )
            )

        except Exception as error:
            await self.callbacks.show(update.effective_message, f"Ошибка генерации:\n{error}")

            return

        decision = getattr(self.reply_service, 'last_decision', None)
        try:
            variants = self.reply_service.split_variants(answers)
        except ValueError as error:
            await self.callbacks.show(update.effective_message, str(error))
            return
        if len(variants) != 3 and not (decision and decision['action'] in {'wait', 'ask_owner'}):
            await self.callbacks.show(update.effective_message,
                "Не удалось получить три отдельных варианта. Нажмите «Ответ» ещё раз.")
            return

        journal_warning = None
        generation_id = None
        try:
            generation_id = record_generation(self.workspace, variants,
                              self.reply_service.last_context_message_id,
                              reply_model(),
                              getattr(self.reply_service, "last_trigger_text", ""),
                              request_snapshot=getattr(self.reply_service, 'last_request_snapshot', None),
                              decision=decision)
        except Exception:
            logging.exception("Не удалось сохранить варианты в журнале диалога")
            journal_warning = (
                "Не удалось сохранить варианты для будущего обучения. "
                "Проверьте журнал программы."
            )

        if journal_warning:
            await self.send_long_text(update.effective_message, journal_warning)
        if decision:
            labels = {'reply': 'Ответить', 'wait': 'Подождать', 'ask_owner': 'Нужно ваше решение'}
            summary = f"{labels[decision['action']]} · {decision['tactic']}"
            if decision['owner_note']:
                summary += '\n' + decision['owner_note']
            await self.send_long_text(update.effective_message, summary,
                reply_markup=review_markup(generation_id) if generation_id else None)
            if decision['action'] != 'reply':
                return
        text = variants_text(variants)
        await self.send_long_text(update.effective_message, text,
            reply_markup=variants_markup(generation_id) if generation_id else None)


    async def reply_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        instruction = None

        if context.args:
            instruction = " ".join(
                context.args
            )

        await self.send_generated_answers(
            update,
            instruction
        )


    async def text_instruction(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        instruction = update.effective_message.text.strip()
        if instruction in PANEL_ACTIONS:
            await self.menu.move_to_bottom()
            await self.callbacks.dispatch(PANEL_ACTIONS[instruction], update, context)
            try:
                await update.effective_message.delete()
            except TelegramError:
                pass
            return
        if await self.reply_practice.handle_text(instruction, update, context):
            return
        if await self.benchmark.handle_text(instruction, update, context):
            return
        if await self.behavior.handle_text(instruction, update, context):
            return
        if await self.goal.handle_text(instruction, update, context):
            return

        search_dialog_id = context.user_data.pop("awaiting_history_search", None)
        if search_dialog_id is not None:
            if self.workspace is None or search_dialog_id != self.workspace.dialog_id:
                await self.callbacks.show(update.effective_message,
                                          "Диалог изменился. Откройте поиск ещё раз.")
                return
            await self.answer_history_question(update, instruction)
            return

        reply_dialog_id = context.user_data.pop("awaiting_reply_instruction", None)
        if reply_dialog_id is not None:
            if self.workspace is None or reply_dialog_id != self.workspace.dialog_id:
                await self.callbacks.show(update.effective_message,
                                          "Диалог изменился. Нажмите «Указания» ещё раз.")
                return
            await self.send_generated_answers(update, instruction)
            return

        if await self.reply_actions.handle_text(instruction, context):
            return

        await self.callbacks.show(update.effective_message,
            "Текст получен. Для вариантов ответа нажмите «✍️ Ответ». "
            "Чтобы дать указания модели, сначала нажмите «📝 Указания». "
            "Для вопроса по переписке используйте «🔎 История».")

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
        await self.runtime_controller.activate(self.workspace.dialog_id,
            lambda text: self.callbacks.show(update.effective_message, text), full_analysis=True)


    async def send_incoming_message(
        self,
        dialog_name,
        message
    ):
        await self.send_live_message(dialog_name, message, outgoing=False)

    async def send_outgoing_message(self, dialog_name, message):
        await self.send_live_message(dialog_name, message, outgoing=True)

    async def send_live_message(self, dialog_name, message, *, outgoing):
        text = message.get(
            "text"
        )

        if not text:
            message_type = (
                message.get(
                    "type",
                    "unknown"
                )
            )

            text = (
                f"[{message_type}]"
            )

        bot_text = (
            f"{'➡️ Вы' if outgoing else '⬅️ ' + dialog_name}:\n{text}"
        )

        for start in range(
            0,
            len(bot_text),
            3000
        ):
            last_chunk = start + 3000 >= len(bot_text)
            markup = None
            if (outgoing and last_chunk and self.workspace is not None
                    and message.get('message_id') is not None
                    and message.get('text')):
                markup = InlineKeyboardMarkup([[
                    InlineKeyboardButton("📝 Разметить ответ",
                        callback_data=(f"ui:fb:{self.workspace.dialog_id}:"
                                       f"{message['message_id']}"))]])
            await self.application.bot.send_message(
                chat_id=self.owner_id,
                text=bot_text[
                    start:
                    start + 3000
                ],
                reply_markup=markup,
            )


    def build_application(
        self
    ):
        if not Config.AGENT_BOT_TOKEN:
            raise ValueError(
                "В .env отсутствует "
                "AGENT_BOT_TOKEN"
            )

        self.application = (
            Application.builder()
            .token(
                Config.AGENT_BOT_TOKEN
            )
            .connect_timeout(30)
            .read_timeout(30)
            .write_timeout(30)
            .get_updates_connect_timeout(30)
            .get_updates_read_timeout(30)
            .build()
        )

        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("style", self.style_command))
        self.application.add_handler(CommandHandler("panel", self.panel_command))
        self.application.add_handler(
            CallbackQueryHandler(self.callbacks.handle, pattern=r"^ui:")
        )

        self.application.add_handler(
            CommandHandler(
                "start",
                self.start_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "dialogs",
                self.dialogs_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "select",
                self.select_dialog_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "show",
                self.show_command
            )
        )

        self.application.add_handler(CommandHandler("search", self.search_command))

        self.application.add_handler(
            CommandHandler(
                "reply",
                self.reply_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "goal",
                self.goal_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "clear_goal",
                self.clear_goal_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "update_memory",
                self.update_memory_command
            )
        )

        self.application.add_handler(
            MessageHandler(
                filters.TEXT
                & ~filters.COMMAND,
                self.text_instruction
            )
        )


    async def start(
        self
    ):
        self.build_application()

        await self.application.initialize()
        self.control_bot_id = self.application.bot.id
        await self.menu.clean_previous_menus()
        await self.panel.install()

        await (
            self.application
            .updater
            .start_polling()
        )

        await self.application.start()
        await self.menu.show(None, self.menu_title(), self.main_menu())

        print(
            "\nTelegram-интерфейс "
            "агента запущен."
        )


    async def stop(
        self
    ):
        if not self.application:
            return

        print(
            "\nОстанавливаю Telegram-бота..."
        )

        if self.application.updater.running:
            await self.application.updater.stop()

        if self.application.running:
            await self.application.stop()
        await self.application.shutdown()

        print(
            "Telegram-бот остановлен."
        )

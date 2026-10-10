from telegram import Update, InlineKeyboardButton
import logging
import asyncio
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
from mybot.telegram.bot_reply_actions import BotReplyActions, variants_markup, variants_text
from mybot.telegram.bot_benchmark import clear_benchmark_input
from mybot.storage.atomic import read_json
from mybot.ai.reply_provider import reply_model
from mybot.telegram.bot_reply_review import review_markup


from mybot.telegram.legacy_interface import LegacyInterface


class BotInterface(LegacyInterface):
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
        from mybot.telegram.bot_simple_lesson import BotSimpleLesson
        self.simple_lesson = BotSimpleLesson(self)
        self.menu = BotMenu(self)
        self.panel = BotPanel(self)
        self.runtime_controller = None
        self.control_bot_id = None


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
        title = f"AI Agent\nАктивный диалог: {active}"
        if self.workspace is not None and hasattr(self.workspace, 'root'):
            state = read_json(self.workspace.root / 'learning' / 'idle_history.json') or {}
            if state.get('status') == 'retry_on_restart':
                title += '\nИстория: автодозагрузка требует проверки.'
            elif state.get('status') == 'recovery_required':
                title += ('\nИстория: найден пропуск. Для восстановления: '
                          'история требует восстановления перед продолжением.')
        return title

    async def start_command(self, update, context):
        if not self.is_owner(update):
            return
        from mybot.telegram.bot_mixed_replies import cancel_clarification
        cancel_clarification(self)
        task = getattr(self, 'mixed_task', None)
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if hasattr(self, 'simple_lesson'):
            self.simple_lesson.leave()
        clear_benchmark_input(context)
        context.user_data.pop("awaiting_mood", None)
        context.user_data.pop("awaiting_goal", None)
        context.user_data.pop("awaiting_behavior", None)
        context.user_data.pop('practice_correction', None)
        context.user_data.pop("pending_reply_edit", None)
        await self.menu.move_to_bottom()
        await self.panel.show()

    def main_menu(self):
        return main_menu(active=self.workspace is not None,
                         panel_hidden=self.panel.hidden)




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


    async def dialogs_command(self, update, context):
        if not self.is_owner(update):
            return
        await self.callbacks.show_dialogs(update.effective_message, context, "ui:dialogs:0")





    @active_operation
    async def send_generated_answers(
        self,
        update,
        instruction=None,
        force_reply=False,
    ):
        if not self.is_owner(
            update
        ):
            return

        await self.callbacks.show(update.effective_message, "Генерирую ответ…")

        try:
            controller = getattr(self, 'runtime_controller', None)
            if controller is not None:
                await controller.refresh_reply_context()
            from mybot.services.mixed_replies import enabled
            if enabled(self.workspace):
                from mybot.telegram.bot_mixed_replies import generate
                await generate(self, update, instruction)
                return
            answers = (
                await self.reply_service.generate(
                    instruction=instruction, allow_decision=not force_reply,
                    force_reply=force_reply,
                )
            )

        except Exception as error:
            rows = None
            if not force_reply and self.workspace is not None:
                from mybot.services.style_versions import selected
                if selected(self.workspace) != 'configured':
                    from mybot.telegram.bot_keyboards import home_menu
                    rows = home_menu([[InlineKeyboardButton('Получить варианты без проверки',
                        callback_data='ui:reply_force')]])
            await self.callbacks.show(update.effective_message, f"Ошибка генерации:\n{error}", rows)

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
                              (getattr(self.reply_service, 'last_request_snapshot', None) or {}).get('model', reply_model()),
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
                reply_markup=(home_menu([[InlineKeyboardButton('Получить варианты всё равно',
                    callback_data='ui:reply_force')]]) if decision['action'] == 'wait'
                    else review_markup(generation_id) if generation_id else None))
            if decision['action'] != 'reply':
                return
        text = variants_text(variants)
        request = getattr(self.reply_service, 'last_request_snapshot', None) or {}
        if request.get('provider') == 'mlx':
            from mybot.services.style_versions import label
            text = label(request['version']) + '\n\n' + text
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
        if await self.simple_lesson.handle_text(instruction, update.effective_message):
            return
        from mybot.telegram.bot_mixed_replies import clarify_text
        if await clarify_text(self, update, instruction):
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
            "Для занятия нажмите «Обучение».")







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

        self.application.add_handler(CommandHandler('start', self.start_command))
        self.application.add_handler(CommandHandler('reply', self.reply_command))
        self.application.add_handler(CommandHandler('dialogs', self.dialogs_command))
        self.application.add_handler(CommandHandler('select', self.select_dialog_command))
        self.application.add_handler(
            CallbackQueryHandler(self.callbacks.handle, pattern=r"^ui:")
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
        await self.menu.move_to_bottom()
        await self.panel.install()

        await (
            self.application
            .updater
            .start_polling()
        )

        await self.application.start()

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

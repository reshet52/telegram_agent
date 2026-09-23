from telegram import Update
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
        self.callbacks = BotCallbacks(self)
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
        chunk_size=3000
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
                ]
            )


    def menu_title(self):
        active = self.workspace.dialog_name if self.workspace else "не выбран"
        return f"AI Agent\nАктивный диалог: {active}"

    async def start_command(self, update, context):
        if not self.is_owner(update):
            return
        context.user_data.pop("awaiting_mood", None)
        await self.panel.show()
        await self.menu.move_to_bottom()
        await self.menu.show(update.effective_message, self.menu_title(), main_menu())

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


    async def mood_command(self, update, context):
        if not self.is_owner(update):
            return
        if context.args:
            self.state.current_mood = " ".join(context.args).strip()
        await self.callbacks.show(update.effective_message,
            f"Настроение: {self.state.current_mood or 'не задано'}")

    async def clear_mood_command(self, update, context):
        if not self.is_owner(update):
            return
        self.state.current_mood = None
        await self.callbacks.show(update.effective_message, "Настроение сброшено.")


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
                    instruction=instruction,
                    current_mood=
                        self.state.current_mood
                )
            )

        except Exception as error:
            await self.callbacks.show(update.effective_message, f"Ошибка генерации:\n{error}")

            return

        variants = (
            self.reply_service.split_variants(
                answers
            )
        )

        await self.menu.show(update.effective_message, self.menu_title(), main_menu())

        for variant in variants:
            await self.send_long_text(
                update.effective_message,
                variant
            )


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
        if context.user_data.pop("awaiting_mood", False):
            self.state.current_mood = instruction
            await self.callbacks.show(update.effective_message, f"Настроение установлено:\n{instruction}")
            return

        await self.send_generated_answers(
            update,
            instruction
        )


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
            f"{dialog_name}:\n{text}"
        )

        for start in range(
            0,
            len(bot_text),
            3000
        ):
            await self.application.bot.send_message(
                chat_id=self.owner_id,
                text=bot_text[
                    start:
                    start + 3000
                ]
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

        self.application.add_handler(
            CommandHandler(
                "reply",
                self.reply_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "mood",
                self.mood_command
            )
        )

        self.application.add_handler(
            CommandHandler(
                "clear_mood",
                self.clear_mood_command
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
        await self.menu.show(None, self.menu_title(), main_menu())

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
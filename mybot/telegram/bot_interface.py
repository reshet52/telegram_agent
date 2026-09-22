from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters
)

from mybot.app.session_state import (
    SessionState
)
from mybot.config import Config
from mybot.memory.incremental import (
    update_incremental_memory
)
from mybot.services.reply_service import (
    ReplyService
)


class BotInterface:
    def __init__(
        self,
        owner_id,
        recent_messages,
        reply_service: ReplyService,
        state: SessionState
    ):
        self.owner_id = owner_id
        self.recent_messages = (
            recent_messages
        )
        self.reply_service = (
            reply_service
        )
        self.state = state

        self.application = None


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


    async def start_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        await update.message.reply_text(
            "AI Agent запущен.\n\n"
            "/reply — предложить ответ\n"
            "/show — текущий диалог\n"
            "/mood — текущее настроение\n"
            "/mood <текст> — задать настроение\n"
            "/clear_mood — сбросить настроение\n"
            "/update_memory — обновить память\n\n"
            "Также можешь просто написать мне "
            "обычным сообщением, что именно "
            "нужно сказать собеседнику."
        )


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
            update.message,
            text
        )


    async def mood_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        if context.args:
            self.state.current_mood = (
                " ".join(
                    context.args
                ).strip()
            )

            await update.message.reply_text(
                "Настроение установлено:\n\n"
                f"{self.state.current_mood}"
            )

            return

        if self.state.current_mood:
            await update.message.reply_text(
                "Текущее настроение:\n\n"
                f"{self.state.current_mood}"
            )

        else:
            await update.message.reply_text(
                "Настроение не задано."
            )


    async def clear_mood_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        self.state.current_mood = None

        await update.message.reply_text(
            "Настроение сброшено."
        )


    async def send_generated_answers(
        self,
        update,
        instruction=None
    ):
        if not self.is_owner(
            update
        ):
            return

        status_message = (
            await update.message.reply_text(
                "Генерирую ответ..."
            )
        )

        try:
            answers = (
                await self.reply_service.generate(
                    instruction=instruction,
                    current_mood=
                        self.state.current_mood
                )
            )

        except Exception as error:
            await status_message.edit_text(
                f"Ошибка генерации:\n{error}"
            )

            return

        variants = (
            self.reply_service.split_variants(
                answers
            )
        )

        await status_message.edit_text(
            f"Вариантов: {len(variants)}"
        )

        for variant in variants:
            await self.send_long_text(
                update.message,
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

        instruction = (
            update.message.text.strip()
        )

        await self.send_generated_answers(
            update,
            instruction
        )


    async def update_memory_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):
        if not self.is_owner(
            update
        ):
            return

        message = (
            await update.message.reply_text(
                "Обновляю память..."
            )
        )

        try:
            result = (
                await update_incremental_memory()
            )

        except Exception as error:
            await message.edit_text(
                "Ошибка обновления памяти:\n"
                f"{error}"
            )

            return

        await message.edit_text(
            "Память обновлена.\n\n"
            f"Новых сообщений: "
            f'{result["messages"]}\n'
            f"Новых memories: "
            f'{result["memories"]}'
        )


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

        self.application.add_handler(
            CommandHandler(
                "start",
                self.start_command
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

        await (
            self.application
            .updater
            .stop()
        )

        await self.application.stop()
        await self.application.shutdown()

        print(
            "Telegram-бот остановлен."
        )
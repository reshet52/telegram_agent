"""One Bot API polling process routes private controls to isolated accounts."""

import asyncio
import getpass
import io
import logging

from telethon import TelegramClient, errors
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from mybot.app.session_state import SessionState
from mybot.config import Config
from mybot.services.runtime_controller import RuntimeController
from mybot.storage.control_state import ControlState
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.bot_keyboards import home_menu


COMMANDS = {
    "start": "start_command", "status": "status_command", "style": "style_command",
    "panel": "panel_command", "dialogs": "dialogs_command", "select": "select_dialog_command",
    "show": "show_command", "reply": "reply_command", "mood": "mood_command",
    "clear_mood": "clear_mood_command", "update_memory": "update_memory_command",
}


class AccountBot:
    def __init__(self, credentials):
        self.credentials = credentials
        self.application = Application.builder().token(Config.AGENT_BOT_TOKEN).build()
        self.clients = {owner: TelegramClient(item.session, item.api_id, item.api_hash)
                        for owner, item in credentials.items()}
        self.interfaces = {}
        self.blocked_owners = set()
        self.login_tasks = {}
        self.login_locks = {owner: asyncio.Lock() for owner in credentials}
        for command in COMMANDS:
            self.application.add_handler(CommandHandler(command, self._command(command)))
        self.application.add_handler(CallbackQueryHandler(self._callback, pattern=r"^(ui:|auth:)") )
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._text))

    def _owner(self, update):
        user, chat = update.effective_user, update.effective_chat
        if not user or not chat or user.id != chat.id or chat.type != "private":
            return None
        return user.id if user.id in self.credentials and user.id not in self.blocked_owners else None

    def _command(self, command):
        async def handler(update, context):
            owner = self._owner(update)
            if owner is None:
                return
            interface = self.interfaces.get(owner)
            if not interface:
                await self._login_screen(owner)
                return
            await getattr(interface, COMMANDS[command])(update, context)
        return handler

    async def _callback(self, update, context):
        owner = self._owner(update)
        if owner is None:
            await update.callback_query.answer()
            return
        data = update.callback_query.data or ""
        if data == "auth:qr":
            await update.callback_query.answer()
            await self._start_qr(owner)
        elif owner in self.interfaces:
            await self.interfaces[owner].callbacks.handle(update, context)
        else:
            await update.callback_query.answer()
            await self._login_screen(owner)

    async def _text(self, update, context):
        owner = self._owner(update)
        if owner is None:
            return
        if owner in self.interfaces:
            await self.interfaces[owner].text_instruction(update, context)
        else:
            await self._login_screen(owner)

    async def _login_screen(self, owner, text=None):
        state = ControlState(owner)
        text = text or "Аккаунт Telegram не авторизован. Войдите по QR-коду."
        markup = InlineKeyboardMarkup([[
            InlineKeyboardButton("Войти по QR", callback_data="auth:qr")]])
        message_id = state.get("menu_message_id")
        if message_id:
            try:
                await self.application.bot.edit_message_text(
                    chat_id=owner, message_id=message_id, text=text, reply_markup=markup)
                return
            except BadRequest as error:
                if "message is not modified" in str(error).lower():
                    return
        try:
            sent = await self.application.bot.send_message(owner, text, reply_markup=markup)
        except BadRequest as error:
            if "chat not found" in str(error).lower():
                # A bot cannot initiate a conversation. /start from this owner
                # will display the same login screen once the chat exists.
                logging.info("Аккаунт %s ещё не открыл чат с управляющим ботом", owner)
                return
            raise
        state.set(menu_message_id=sent.message_id)

    async def _verify(self, owner):
        client = self.clients[owner]
        await client.connect()
        if not await client.is_user_authorized():
            return None
        me = await client.get_me()
        if not me or me.id != owner:
            await client.disconnect()
            raise ValueError(f"Сессия {owner} принадлежит другому Telegram ID; она не будет использована.")
        return me

    async def _ready(self, owner, me):
        client = self.clients[owner]
        interface = BotInterface(owner, [], None, SessionState(), telegram_client=client)
        interface.control_state = ControlState(owner)
        interface.panel.hidden = bool(interface.control_state.get("panel_hidden", False))
        interface.application = self.application
        interface.control_bot_id = self.application.bot.id
        controller = RuntimeController(client, me, interface)
        interface.runtime_controller = controller
        self.interfaces[owner] = interface
        saved_id = interface.control_state.get("active_dialog_id")
        message = None
        markup = None
        if saved_id is not None:
            try:
                await controller.restore(int(saved_id))
            except Exception as error:
                message = f"Не удалось восстановить диалог: {error}\nДанные сохранены. Откройте «Диалоги» или «Статус»."
                markup = home_menu([[
                    InlineKeyboardButton("💬 Диалоги", callback_data="ui:dialogs:0")], [
                    InlineKeyboardButton("⏳ Статус", callback_data="ui:status")]])
                logging.exception("Не удалось восстановить диалог аккаунта %s", owner)
        elif controller.job.get("status") == "paused":
            message = controller.last_status + "\n\nПодготовка приостановлена. Продолжение только по кнопке."
            markup = home_menu([[
                InlineKeyboardButton("▶️ Продолжить", callback_data="ui:resume")]])
        try:
            await interface.menu.clean_previous_menus()
            await interface.panel.install()
            await interface.menu.show(None, message or interface.menu_title(),
                                      markup or interface.main_menu())
        except Exception:
            logging.exception("Не удалось восстановить меню для %s", owner)
        return interface

    async def _start_qr(self, owner):
        if owner in self.interfaces:
            return
        task = self.login_tasks.get(owner)
        if task and not task.done():
            await self.application.bot.send_message(owner, "QR-вход уже ожидает подтверждения.")
            return
        self.login_tasks[owner] = asyncio.create_task(self._qr_login(owner))

    async def _qr_login(self, owner):
        async with self.login_locks[owner]:
            client = self.clients[owner]
            try:
                import qrcode
                await client.connect()
                for _ in range(3):
                    qr = await client.qr_login()
                    image = qrcode.make(qr.url)
                    payload = io.BytesIO()
                    image.save(payload, format="PNG")
                    payload.seek(0)
                    sent = await self.application.bot.send_photo(owner, payload,
                        caption="Откройте Telegram → Настройки → Устройства → Подключить устройство и отсканируйте QR.")
                    try:
                        me = await qr.wait(timeout=90)
                        break
                    except asyncio.TimeoutError:
                        await sent.delete()
                        continue
                    except errors.SessionPasswordNeededError:
                        password = await asyncio.to_thread(getpass.getpass,
                            f"Telegram 2FA для {owner} (локальная консоль): ")
                        me = await client.sign_in(password=password)
                        break
                else:
                    await self._login_screen(owner, "Срок действия QR истёк. Нажмите «Войти по QR» снова.")
                    return
                try:
                    await sent.delete()
                except Exception:
                    logging.warning("Не удалось удалить использованный QR для %s", owner)
                if me.id != owner:
                    await client.log_out()
                    await self._login_screen(owner, "QR подтвердил другой аккаунт. Войдите нужным аккаунтом.")
                    return
                await self._ready(owner, me)
            except Exception as error:
                logging.exception("QR-вход не удался для %s", owner)
                await self._login_screen(owner, f"Не удалось войти: {error}")

    async def start(self):
        await self.application.initialize()
        await self.application.bot.set_my_commands([
            BotCommand("start", "Пульт управления"), BotCommand("dialogs", "Выбрать диалог"),
            BotCommand("reply", "Предложить ответ"), BotCommand("status", "Прогресс подготовки"),
            BotCommand("panel", "Показать пульт")])
        await self.application.updater.start_polling()
        await self.application.start()
        for owner in self.credentials:
            try:
                me = await self._verify(owner)
            except ValueError as error:
                self.blocked_owners.add(owner)
                logging.error("Аккаунт %s отключён: %s. Исправьте локальный список ID.", owner, error)
                continue
            except Exception:
                logging.exception("Не удалось проверить сессию аккаунта %s", owner)
                continue
            try:
                if me:
                    await self._ready(owner, me)
                else:
                    await self._login_screen(owner)
            except Exception:
                logging.exception("Не удалось запустить интерфейс аккаунта %s", owner)

    async def stop(self):
        for task in self.login_tasks.values():
            task.cancel()
        if self.login_tasks:
            await asyncio.gather(*self.login_tasks.values(), return_exceptions=True)
        for interface in self.interfaces.values():
            await interface.runtime_controller.stop()
        for client in self.clients.values():
            await client.disconnect()
        if self.application.updater.running:
            await self.application.updater.stop()
        if self.application.running:
            await self.application.stop()
        await self.application.shutdown()

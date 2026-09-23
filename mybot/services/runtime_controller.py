"""One active dialog, one listener; preparation is explicitly confirmed in Telegram."""

import asyncio
import logging
import time

from mybot.services.dialog_runtime import prepare_dialog_runtime
from mybot.services.typing_service import TypingService
from mybot.storage.workspace import get_workspace_root, get_global_root
from mybot.storage.atomic import read_json, write_json
from mybot.storage.reply_journal import record_outgoing_reply
from mybot.telegram.live_events import LiveEvents
from mybot.telegram.exporter import fetch_messages_after, append_message_data
from mybot.storage.history import get_last_saved_message_id
from mybot.episodes.incremental import process_live_episode_message


class RuntimeController:
    def __init__(self, client, me, bot):
        self.client = client
        self.me = me
        self.bot = bot
        self.typing = TypingService(client)
        self.runtime = None
        self.listener = None
        self.task = None
        self.lock = asyncio.Lock()
        self.busy = False
        self.job_path = get_global_root(me.id) / 'preparation_job.json'
        self.job = read_json(self.job_path, {})
        if self.job.get('status') == 'running':
            self.job['status'] = 'paused'
        self.last_status = self.job.get('progress', 'Подготовка не запущена.')

    async def describe(self, dialog_id):
        dialog = await self.bot.agent_manager.select_dialog(dialog_id)
        if dialog is None:
            raise ValueError("Диалог больше недоступен.")
        if dialog_id == self.bot.control_bot_id:
            raise ValueError("Управляющего бота нельзя выбрать как рабочий диалог.")
        root = get_workspace_root(self.me.id, dialog_id)
        exists = (root / "chat_history.jsonl").exists()
        info = await self.client.get_messages(dialog_id, limit=0)
        kind = "Существующий диалог" if exists else "Новый диалог"
        return (
            f"{dialog.name or dialog.id}\nID: {dialog.id}\n"
            f"{kind}. Сообщений в Telegram: {info.total or 0}.\n\n"
            "История сохраняется в отдельный файл этого диалога.\n"
            "Для нового диалога: эпизоды → анализ всей истории → память → индексы → общий стиль.\n"
            f"Ориентир: от {(info.total or 0) // 100 + bool((info.total or 0) % 100)} частей анализа "
            "по 100 сообщений; длинные сообщения делятся дополнительно. "
            "Также выполняются запросы объединения памяти, embeddings и анализа стиля.\n"
            "Это платные запросы API; точная сумма зависит от объёма текста. "
            "Готовая память не анализируется заново без отдельного запуска.\n\n"
            "Начать подготовку и открыть диалог?"
        )

    async def activate(self, dialog_id, progress, full_analysis=False):
        if self.busy or self.lock.locked():
            await progress("Дождитесь завершения текущей операции.")
            return
        self.busy = True
        self.job = {'dialog_id': dialog_id, 'full_analysis': full_analysis, 'status': 'running'}
        write_json(self.job_path, self.job)
        self.task = asyncio.create_task(self._activate(dialog_id, progress, full_analysis))

    async def _activate(self, dialog_id, progress, full_analysis=False):
        old_listener = self.listener
        report = progress
        last_report = 0.0

        async def progress(text):
            nonlocal last_report
            self.last_status = text
            self.job['progress'] = text
            write_json(self.job_path, self.job)
            now = time.monotonic()
            if self.job.get('status') == 'running' and now - last_report < 1.5:
                return
            last_report = now
            try:
                await report(text)
            except Exception:
                logging.exception("Не удалось обновить меню подготовки")

        try:
            async with self.lock:
                await self.typing.stop()
                dialog = await self.bot.agent_manager.find_dialog(dialog_id)
                if dialog is None or dialog_id == self.bot.control_bot_id:
                    raise ValueError("Диалог недоступен.")
                if old_listener:
                    await old_listener.unregister()
                runtime = await prepare_dialog_runtime(
                    self.client, dialog, self.me, self.bot.state,
                    bot_interface=self.bot, progress=progress, full_analysis=full_analysis,
                )
                listener = LiveEvents(
                    client=self.client, selected_dialog=dialog, me=self.me,
                    dialog_name=runtime.dialog_name, workspace=runtime.workspace,
                    raw_history=runtime.raw_history,
                    recent_messages=runtime.recent_messages,
                    known_message_ids=runtime.known_message_ids,
                    bot_interface=self.bot, episode_tracker=runtime.episode_tracker,
                    typing_service=self.typing,
                )
                # Register before catching up the preparation window. The listener
                # lock and known IDs serialize and deduplicate overlapping events.
                listener.register()
                try:
                    async with listener.message_processing_lock:
                        missed = await fetch_messages_after(
                            client=self.client, dialog=dialog,
                            min_message_id=get_last_saved_message_id(runtime.workspace.chat_history),
                            me_id=self.me.id,
                        )
                        for message in missed:
                            message_id = message.get("message_id")
                            if message_id in runtime.known_message_ids:
                                continue
                            if message.get("sender") == "Я":
                                try:
                                    record_outgoing_reply(runtime.workspace, message)
                                except Exception:
                                    logging.exception("Не удалось сохранить исходящее сообщение в журнале")
                            append_message_data(message, filename=runtime.workspace.chat_history)
                            runtime.known_message_ids.add(message_id)
                            runtime.recent_messages.append(message)
                            del runtime.recent_messages[:-15]
                            await process_live_episode_message(
                                runtime.episode_tracker, message,
                                episodes_filename=runtime.workspace.episodes,
                                live_embeddings_filename=runtime.workspace.episode_embeddings_live,
                            )
                except BaseException:
                    await listener.unregister()
                    raise
                self.runtime = runtime
                self.listener = listener
                self.bot.workspace = runtime.workspace
                self.bot.recent_messages = runtime.recent_messages
                self.bot.reply_service = runtime.reply_service
        except asyncio.CancelledError:
            if old_listener:
                await old_listener.unregister()
            self.clear_runtime()
            self.job['status'] = 'paused'
            await progress('Подготовка приостановлена. Сохранённые этапы будут использованы при продолжении.')
            raise
        except Exception as error:
            if old_listener:
                await old_listener.unregister()
            self.clear_runtime()
            self.job["status"] = "failed"
            await progress(f"Не удалось открыть диалог:\n{error}\n"
                           "Активный диалог отключён. Повторите открытие после устранения ошибки.")
        else:
            self.job["status"] = "ready"
            if self.bot.application:
                try:
                    await self.bot.panel.refresh()
                except Exception:
                    logging.exception("Не удалось обновить пульт диалога")
            await progress(f"Активный диалог: {runtime.dialog_name}\n"
                           "Можно смотреть контекст и создавать ответы.")
        finally:
            self.busy = False

    def clear_runtime(self):
        self.listener = None
        self.runtime = None
        self.bot.workspace = None
        self.bot.reply_service = None
        self.bot.recent_messages = []

    async def pause(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            if self.busy:  # Cancelled before the background coroutine started.
                await self.typing.stop()
                if self.listener:
                    await self.listener.unregister()
                self.clear_runtime()
                self.busy = False
                self.job['status'] = 'paused'
                self.last_status = 'Подготовка приостановлена.'
                self.job['progress'] = self.last_status
                write_json(self.job_path, self.job)

    async def resume(self, progress):
        if self.job.get('status') not in {'paused', 'failed'}:
            await progress('Нет приостановленной подготовки.')
            return
        await self.activate(self.job['dialog_id'], progress, self.job.get('full_analysis', False))

    async def stop(self):
        await self.typing.stop()
        await self.pause()
        if self.listener:
            await self.listener.unregister()

    async def deactivate(self):
        """Leave the active dialog and remove its listener without discarding workspace data."""
        if self.busy or self.lock.locked():
            return False
        async with self.lock:
            await self.typing.stop()
            if self.listener:
                await self.listener.unregister()
            self.clear_runtime()
            self.bot.agent_manager.selected_dialog = None
        if self.bot.application:
            try:
                await self.bot.panel.refresh()
            except Exception:
                logging.exception("Не удалось обновить общий пульт")
        return True

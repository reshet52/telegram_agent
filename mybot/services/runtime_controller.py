"""One active dialog, one listener; preparation is explicitly confirmed in Telegram."""

import asyncio
import logging
import time

from mybot.services.dialog_runtime import prepare_dialog_runtime, restore_ready_runtime, reply_ready
from mybot.storage.workspace import get_workspace_root, get_global_root
from mybot.storage.atomic import read_json, write_json
from mybot.storage.atomic import durable_call
from mybot.storage.reply_journal import record_outgoing_reply
from mybot.telegram.live_events import LiveEvents
from mybot.telegram.exporter import fetch_messages_after, append_message_data
from mybot.storage.history import get_last_saved_message_id
from mybot.episodes.incremental import process_live_episode_message
from mybot.services.history_import import commit_batch as commit_history_batch
from mybot.services.workspace_compaction import compact_workspace
from mybot.services.history_update import stage_history_update
from mybot.services.workspace_initializer import is_prepared, has_legacy_memory
from mybot.storage.workspace import Workspace
from mybot.storage.history_transaction import MARKER as HISTORY_UPDATE_MARKER
from mybot.services.idle_history import IdleHistory


class RuntimeController:
    def __init__(self, client, me, bot):
        self.client = client
        self.me = me
        self.bot = bot
        self.runtime = None
        self.listener = None
        self.task = None
        self.finish_task = None
        self.update_task = None
        self.update_committed = False
        self.lock = asyncio.Lock()
        self.busy = False
        self.idle_history = IdleHistory(self)
        self.online_task = None
        self.job_path = get_global_root(me.id) / 'preparation_job.json'
        self.job = read_json(self.job_path, {})
        if self.job.get('status') == 'running':
            self.job['status'] = 'paused'
            write_json(self.job_path, self.job)
        self.last_status = self.job.get('progress', 'Подготовка не запущена.')

    @property
    def can_cancel(self):
        if not self.busy:
            return False
        if self.update_task and not self.update_task.done():
            return not self.update_committed
        if self.finish_task and not self.finish_task.done():
            return True
        return self.job.get('status') == 'running'

    async def describe(self, dialog_id):
        dialog = await self.bot.agent_manager.select_dialog(dialog_id)
        if dialog is None:
            raise ValueError("Диалог больше недоступен.")
        if dialog_id == self.bot.control_bot_id:
            raise ValueError("Управляющего бота нельзя выбрать как рабочий диалог.")
        root = get_workspace_root(self.me.id, dialog_id)
        exists = (root / "chat_history.jsonl").exists()
        workspace = Workspace(self.me.id, dialog_id, dialog.name, root)
        if exists and (reply_ready(workspace) or is_prepared(workspace) or has_legacy_memory(workspace)):
            return (f"{dialog.name or dialog.id}\nСуществующий диалог. История и память сохранены.\n\n"
                    "При открытии будут загружены новые сообщения и добавлены новые эпизоды. "
                    "Полный анализ истории не запускается. Открыть диалог?")
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
        if self.online_task is None or self.online_task.done():
            from mybot.services.online_learning import monitor
            self.online_task = asyncio.create_task(monitor(self))
        if self.busy or self.lock.locked():
            await progress("Дождитесь завершения текущей операции.")
            return
        self.busy = True
        self.job = {'dialog_id': dialog_id, 'full_analysis': full_analysis, 'status': 'running'}
        write_json(self.job_path, self.job)
        self.task = asyncio.create_task(self._activate(dialog_id, progress, full_analysis))

    async def _activate(self, dialog_id, progress, full_analysis=False):
        old_listener = self.listener
        listener = None
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
                            if runtime.episode_tracker is not None:
                                await process_live_episode_message(
                                    runtime.episode_tracker, message,
                                    episodes_filename=runtime.workspace.episodes,
                                    live_embeddings_filename=runtime.workspace.episode_embeddings_live,
                                )
                except BaseException:
                    await listener.unregister()
                    raise
                if getattr(self.bot, "control_state", None):
                    self.bot.control_state.set(active_dialog_id=dialog_id)
                self.runtime = runtime
                self.listener = listener
                self.bot.workspace = runtime.workspace
                self.bot.recent_messages = runtime.recent_messages
                self.bot.reply_service = runtime.reply_service
                self.idle_history.stop()
                self.idle_history.schedule(listener, {}, resumed=True)
        except asyncio.CancelledError:
            if listener:
                await listener.unregister()
            if old_listener:
                await old_listener.unregister()
            self.clear_runtime()
            if getattr(self.bot, "control_state", None):
                try:
                    self.bot.control_state.set(active_dialog_id=None)
                except Exception:
                    logging.exception("Не удалось очистить состояние активного диалога")
            self.job['status'] = 'paused'
            await progress('Подготовка приостановлена. Сохранённые этапы будут использованы при продолжении.')
            raise
        except Exception as error:
            if listener:
                await listener.unregister()
            if old_listener:
                await old_listener.unregister()
            self.clear_runtime()
            if getattr(self.bot, "control_state", None):
                try:
                    self.bot.control_state.set(active_dialog_id=None)
                except Exception:
                    logging.exception("Не удалось очистить состояние активного диалога")
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
        from mybot.ai.local_worker import worker
        try:
            asyncio.get_running_loop().create_task(worker.release())
        except RuntimeError:
            pass
        from mybot.telegram.bot_mixed_replies import cancel_clarification
        cancel_clarification(self.bot)
        if hasattr(self.bot, 'simple_lesson'):
            self.bot.simple_lesson.leave()
        mixed = getattr(self.bot, 'mixed_task', None)
        if mixed and not mixed.done():
            mixed.cancel()
        if self.runtime and getattr(getattr(self.runtime, 'workspace', None), 'root', None):
            write_json(self.runtime.workspace.root / "learning/online_pause.json", {"pause": True})
        self.idle_history.stop()
        self.listener = None
        self.runtime = None
        self.bot.workspace = None
        self.bot.reply_service = None
        self.bot.recent_messages = []

    async def pause(self):
        if self.update_task and not self.update_task.done():
            if self.update_committed:
                return  # Publication completed; don't claim it was rolled back.
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
            self.busy = False
            self.last_status = 'Обновление отменено. Изменения обновления не применены.'
            return
        if self.finish_task and not self.finish_task.done():
            self.finish_task.cancel()
            try:
                await self.finish_task
            except asyncio.CancelledError:
                pass
            # A task cancelled before its first instruction has no finally block.
            self.busy = False
            self.last_status = ('Обновление истории отменено. Сохранённые данные оставлены. '
                                'Если диалог закрыт, откройте его снова для загрузки пропущенных сообщений.')
            return
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            if self.busy:  # Cancelled before the background coroutine started.
                if self.listener:
                    await self.listener.unregister()
                self.clear_runtime()
                if getattr(self.bot, "control_state", None):
                    self.bot.control_state.set(active_dialog_id=None)
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
        if hasattr(self.bot, 'simple_lesson'):
            self.bot.simple_lesson.leave()
            if self.bot.simple_lesson.task:
                await asyncio.gather(self.bot.simple_lesson.task, return_exceptions=True)
        if self.online_task:
            self.online_task.cancel()
        mixed = getattr(self.bot, 'mixed_task', None)
        if mixed and not mixed.done():
            mixed.cancel()
            await asyncio.gather(mixed, return_exceptions=True)
        if self.runtime and getattr(getattr(self.runtime, 'workspace', None), 'root', None):
            write_json(self.runtime.workspace.root / "learning/online_pause.json", {"pause": True})
        await self.pause()
        if self.listener:
            await self.listener.unregister()

    async def finish_dialog(self, report):
        """Explicitly sync messages, compact live files, and leave this dialog."""
        if self.busy or self.lock.locked() or self.runtime is None or self.listener is None:
            return False
        self.busy = True
        self.finish_task = asyncio.create_task(self._finish_dialog(report))
        return True

    async def update_history(self, report):
        if self.busy or self.lock.locked() or self.runtime is None or self.listener is None:
            return False
        self.busy = True
        self.update_committed = False
        self.update_task = asyncio.create_task(self._update_history(report))
        return True

    async def refresh_reply_context(self):
        """Read Telegram's tail; caller holds controller.lock via active_operation."""
        from mybot.telegram.exporter import message_to_data
        from mybot.services.message_order import ordered_messages
        runtime, listener = self.runtime, self.listener
        if runtime is None or listener is None:
            raise ValueError('Сначала откройте диалог.')
        async with listener.message_processing_lock:
            if self.runtime is not runtime or self.listener is not listener:
                raise ValueError('Активный диалог изменился. Повторите запрос.')
            tail = await self.client.get_messages(listener.selected_dialog.id, limit=15)
            messages = [await message_to_data(item, self.me.id, listener.selected_dialog.name)
                        for item in tail]
            if not messages:
                raise ValueError('Telegram не вернул сообщения для ответа.')
            # Mutate the shared list; ReplyService and LiveEvents retain its identity.
            runtime.recent_messages[:] = ordered_messages(messages)

    async def _update_history(self, report):
        transaction = None
        runtime, listener = self.runtime, self.listener

        async def notify(text, done=False):
            self.last_status = text
            try:
                await report(text, done)
            except Exception:
                logging.exception('Не удалось показать прогресс обновления истории')

        try:
            async with self.lock, listener.message_processing_lock:
                transaction, tracker, missing, count = await stage_history_update(
                    self.client, self.me.id, listener.selected_dialog, runtime.workspace, notify)
                transaction.publish()
                self.update_committed = True
                runtime.episode_tracker = listener.episode_tracker = tracker
                runtime.known_message_ids.update(item['message_id'] for item in missing)
                runtime.raw_history.extend(missing)
                runtime.recent_messages.extend(missing)
                from mybot.services.message_order import ordered_messages
                runtime.raw_history[:] = ordered_messages(runtime.raw_history)
                runtime.recent_messages[:] = ordered_messages(runtime.recent_messages)[-15:]
                idle_path = self.idle_history.path(runtime.workspace)
                idle_state = read_json(idle_path, {})
                if idle_state.get('status') == 'recovery_required':
                    idle_state.update(status='synced', verified_id=get_last_saved_message_id(runtime.workspace.chat_history))
                    idle_state.pop('error', None)
                    write_json(idle_path, idle_state)
                for item in missing:
                    if item.get('sender') == 'Я':
                        try:
                            record_outgoing_reply(runtime.workspace, item)
                        except Exception:
                            logging.exception('Не удалось обновить журнал исходящих сообщений')
            await notify(f'Новая история обновлена: сообщений {len(missing)}, эпизодов {count}. '
                         'Диалог остаётся открытым.', True)
        except asyncio.CancelledError:
            if not self.update_committed:
                self.last_status = 'Обновление отменено. Изменения обновления не применены.'
            raise
        except Exception as error:
            logging.exception('Не удалось обновить новую историю')
            if (runtime.workspace.root / HISTORY_UPDATE_MARKER).exists():
                # Rollback itself failed (e.g. disk unavailable). Do not allow
                # live messages to overwrite the files needed for recovery.
                await listener.unregister()
                self.clear_runtime()
                if getattr(self.bot, 'control_state', None):
                    self.bot.control_state.set(active_dialog_id=None)
                await notify('Восстановление файлов не завершено. Диалог отключён; '
                             'резервная копия сохранена. Устраните ошибку диска и откройте диалог заново.', True)
                return
            await notify(f'Обновление не применено: {error}', True)
        finally:
            if transaction:
                transaction.close()
            self.busy = False

    async def _finish_dialog(self, report):
        listener = self.listener
        runtime = self.runtime
        detached = False

        async def notify(text, done=False):
            self.last_status = text
            try:
                await report(text, done)
            except Exception:
                logging.exception("Не удалось обновить меню завершения диалога")

        try:
            async with self.lock:
                detached = True
                await listener.unregister()  # Drains in-flight message writes.
                last_id = get_last_saved_message_id(runtime.workspace.chat_history)
                if not last_id:
                    raise ValueError("Нет сохранённого ID истории. Полная загрузка при обновлении запрещена.")
                missed = await fetch_messages_after(
                    self.client, listener.selected_dialog,
                    last_id, self.me.id)
                missing = [item for item in missed
                           if item.get("message_id") not in runtime.known_message_ids]
                if missing:
                    await durable_call(commit_history_batch, runtime.workspace.chat_history, missing)
                    for item in missing:
                        if item.get("sender") == "Я":
                            try:
                                record_outgoing_reply(runtime.workspace, item)
                            except Exception:
                                logging.exception("Не удалось обновить журнал ответов")
                        runtime.known_message_ids.add(item.get("message_id"))
                        runtime.raw_history.append(item)
                        runtime.recent_messages.append(item)
                    del runtime.recent_messages[:-15]
                await notify(f"История: добавлено {len(missing)} сообщений. Объединяю live-данные…")
                merged = await durable_call(compact_workspace, runtime.workspace)
                if getattr(self.bot, "control_state", None):
                    self.bot.control_state.set(active_dialog_id=None)
                self.clear_runtime()
                self.bot.agent_manager.selected_dialog = None
                detached = False
            if self.bot.application:
                try:
                    await self.bot.panel.refresh()
                except Exception:
                    logging.exception("Не удалось обновить пульт после завершения диалога")
            await notify(
                f"История обновлена: новых сообщений {len(missing)}; "
                f"embeddings эпизодов {merged['episode_embeddings']}; "
                f"записей памяти {merged['memories']}. Диалог завершён.", True)
        except asyncio.CancelledError:
            if detached:
                # Do not reattach before catching up: a newer live ID would hide
                # unsaved messages from the cancelled fetch on the next import.
                await listener.unregister()
                if getattr(self.bot, "control_state", None):
                    self.bot.control_state.set(active_dialog_id=None)
                self.clear_runtime()
                self.bot.agent_manager.selected_dialog = None
            await notify("Обновление истории отменено. Сохранённые данные оставлены. "
                         "Откройте диалог снова, чтобы догрузить пропущенные сообщения.", True)
            raise
        except Exception as error:
            logging.exception("Не удалось завершить диалог")
            if detached:
                try:
                    listener.register()
                except Exception:
                    logging.exception("Не удалось вернуть слушатель после ошибки")
            await notify(f"Не удалось обновить историю: {error}\n"
                         "Исходные live-файлы сохранены; можно повторить попытку.", True)
        finally:
            self.busy = False

    async def deactivate(self):
        """Leave the active dialog and remove its listener without discarding workspace data."""
        if self.busy or self.lock.locked():
            return False
        async with self.lock:
            if getattr(self.bot, "control_state", None):
                self.bot.control_state.set(active_dialog_id=None)
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

    async def restore(self, dialog_id):
        """Reattach one ready dialog without triggering paid preparation."""
        if self.online_task is None or self.online_task.done():
            from mybot.services.online_learning import monitor
            self.online_task = asyncio.create_task(monitor(self))
        async with self.lock:
            dialog = await self.bot.agent_manager.find_dialog(dialog_id)
            if dialog is None or dialog.id == self.bot.control_bot_id:
                raise ValueError("Сохранённый диалог больше не доступен. Выберите другой через «Диалоги».")
            runtime = restore_ready_runtime(dialog, self.me, self.bot)
            listener = LiveEvents(client=self.client, selected_dialog=dialog, me=self.me,
                dialog_name=runtime.dialog_name, workspace=runtime.workspace,
                raw_history=runtime.raw_history, recent_messages=runtime.recent_messages,
                known_message_ids=runtime.known_message_ids, bot_interface=self.bot,
                episode_tracker=runtime.episode_tracker)
            listener.register()
            self.runtime, self.listener = runtime, listener
            self.bot.workspace = runtime.workspace
            self.bot.recent_messages = runtime.recent_messages
            self.bot.reply_service = runtime.reply_service
            self.idle_history.schedule(listener, {}, resumed=True)
            self.bot.agent_manager.selected_dialog = dialog
            return runtime

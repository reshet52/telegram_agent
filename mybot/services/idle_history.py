"""Debounced, account/dialog-scoped catch-up without episode or ML training work."""

import asyncio
import logging
from datetime import datetime, timezone

from mybot.storage.atomic import durable_call, read_json, write_json
from mybot.storage.history import get_last_saved_message_id, load_all_messages
from mybot.services.history_import import (HistoryRecoveryRequired, commit_batch,
                                            merge_out_of_order_batch)
from mybot.telegram.exporter import fetch_messages_after
from mybot.services.message_order import ordered_messages

SYNC_AFTER = 30 * 60


class IdleHistory:
    def __init__(self, controller):
        self.controller = controller
        self.task = None

    def path(self, workspace):
        return workspace.root / 'learning' / 'idle_history.json'

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
        self.task = None

    def schedule(self, listener, message, *, resumed=False):
        controller = self.controller
        if controller.listener is not listener or controller.runtime is None:
            return
        workspace = getattr(listener, 'workspace', None)
        if workspace is None or not hasattr(workspace, 'root'):
            return
        if resumed:
            state = read_json(self.path(workspace)) or {}
            if (state.get('account_id'), state.get('dialog_id')) != (workspace.account_id, workspace.dialog_id):
                return
            if state.get('status') in {'synced', 'recovery_required'}:
                return
        else:
            old = read_json(self.path(workspace)) or {}
            if old.get('status') == 'recovery_required':
                self.stop()
                return
            verified = old.get('verified_id')
            if type(verified) is not int:
                verified = get_last_saved_message_id(workspace.chat_history)
            state = {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
                     'message_id': message['message_id'], 'at': datetime.now(timezone.utc).isoformat(),
                     'verified_id': verified, 'status': 'pending'}
            write_json(self.path(workspace), state)
        self.stop()
        self.task = asyncio.create_task(self._run(listener, state))

    async def _run(self, listener, state):
        workspace = listener.workspace
        try:
            started = datetime.fromisoformat(state['at'])
            await asyncio.sleep(max(0, SYNC_AFTER - (datetime.now(timezone.utc) - started).total_seconds()))
            if not self._current(listener, state):
                return
            await self._sync(listener, state)
            state['status'] = 'synced'
            write_json(self.path(workspace), state)
        except asyncio.CancelledError:
            raise
        except HistoryRecoveryRequired as error:
            logging.error('Автодозагрузка истории остановлена: %s', error)
            if self._current(listener, state):
                state['status'] = 'recovery_required'
                state['error'] = str(error)[:300]
                write_json(self.path(workspace), state)
        except Exception as error:
            logging.exception('Не удалось дозагрузить историю после паузы')
            if self._current(listener, state):
                state['status'] = 'retry_on_restart'
                state['error'] = str(error)[:300]
                write_json(self.path(workspace), state)

    def _current(self, listener, state):
        controller = self.controller
        latest = read_json(self.path(listener.workspace)) or {}
        return (controller.listener is listener and controller.runtime is not None
                and latest.get('message_id') == state.get('message_id')
                and latest.get('at') == state.get('at'))

    async def _sync(self, listener, state):
        controller = self.controller
        async with controller.lock, listener.message_processing_lock:
            if controller.listener is not listener:
                return
            workspace = listener.workspace
            last_id = state.get('verified_id')
            if not last_id:
                raise ValueError('Нет сохранённого ID истории; дозагрузка не запускается.')
            fetched = await fetch_messages_after(controller.client, listener.selected_dialog,
                                                 last_id, controller.me.id)
            saved_ids = {m['message_id'] for m in await asyncio.to_thread(
                load_all_messages, workspace.chat_history, include_deleted=True)}
            missing = [m for m in fetched if m['message_id'] not in saved_ids]
            if missing:
                if min(m['message_id'] for m in missing) < max(saved_ids, default=0):
                    await durable_call(merge_out_of_order_batch, workspace, missing)
                else:
                    await durable_call(commit_batch, workspace.chat_history, missing)
                from mybot.storage.reply_journal import record_outgoing_reply
                for item in missing:
                    listener.known_message_ids.add(item['message_id'])
                    listener.raw_history.append(item)
                    listener.recent_messages.append(item)
                    if item.get('sender') == 'Я':
                        try:
                            record_outgoing_reply(workspace, item)
                        except Exception:
                            logging.exception('Не удалось обновить журнал исходящих сообщений')
                listener.raw_history[:] = ordered_messages(listener.raw_history)
                listener.recent_messages[:] = ordered_messages(listener.recent_messages)[-listener.recent_messages_limit:]
            state['verified_id'] = max([last_id] + [m['message_id'] for m in fetched])

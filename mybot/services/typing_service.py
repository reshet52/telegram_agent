"""Explicit, temporary typing activity from the account in one captured dialog."""

import asyncio

from telethon import functions, types


class TypingService:
    def __init__(self, client, *, duration=300, interval=4, think_delay=8,
                 request_timeout=10):
        self.client = client
        self.duration = duration
        self.interval = interval
        self.think_delay = think_delay
        self.request_timeout = request_timeout
        self.dialog_id = None
        self.phase = None
        self.task = None
        self.last_error = None
        self._peer = None
        self._lock = asyncio.Lock()

    async def _send(self, peer, action):
        await asyncio.wait_for(self.client(
            functions.messages.SetTypingRequest(peer=peer, action=action)),
            timeout=self.request_timeout)

    async def _cancel(self, peer):
        try:
            await self._send(peer, types.SendMessageCancelAction())
        except Exception:
            # If offline, Telegram expires the last activity without renewals.
            self.last_error = "Не удалось подтвердить остановку в Telegram; индикатор погаснет сам."

    async def start(self, dialog_id, *, restart=False):
        async with self._lock:
            await self._start_locked(dialog_id, restart=restart)

    async def _start_locked(self, dialog_id, *, restart=False):
        if (not restart and self.dialog_id == dialog_id
                and self.task and not self.task.done()):
            return
        await self._stop()
        self.last_error = None
        peer = await asyncio.wait_for(self.client.get_input_entity(dialog_id),
                                      timeout=self.request_timeout)
        try:
            acknowledged = await asyncio.wait_for(
                self.client.send_read_acknowledge(peer), timeout=self.request_timeout)
        except Exception:
            self.last_error = "Не удалось отметить сообщения прочитанными. Печать не началась."
            raise
        if acknowledged is False:
            self.last_error = "Telegram не подтвердил прочтение. Печать не началась."
            raise RuntimeError(self.last_error)
        self.dialog_id = dialog_id
        self.phase = "thinking"
        self._peer = peer
        self.task = asyncio.create_task(self._run(peer))

    async def on_incoming(self, dialog_id):
        """Re-read the active chat and restart the thinking pause after new input."""
        async with self._lock:
            if self.dialog_id == dialog_id and self.task and not self.task.done():
                await self._start_locked(dialog_id, restart=True)

    async def _run(self, peer):
        deadline = asyncio.timeout(self.duration)
        try:
            await asyncio.sleep(self.think_delay)
            await self._send(peer, types.SendMessageTypingAction())
            self.phase = "typing"
            async with deadline:
                while True:
                    await asyncio.sleep(self.interval)
                    await self._send(peer, types.SendMessageTypingAction())
        except TimeoutError:
            if not deadline.expired():
                self.last_error = "Telegram не ответил вовремя. Индикатор остановлен."
        except asyncio.CancelledError:
            raise
        except Exception:
            self.last_error = "Telegram перестал принимать индикатор. Можно попробовать включить его снова."
        finally:
            await self._cancel(peer)
            self.dialog_id = None
            self.phase = None
            self._peer = None

    async def _stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        # A task cancelled before its first iteration does not run its finally.
        if self._peer is not None:
            await self._cancel(self._peer)
        self.dialog_id = None
        self.phase = None
        self._peer = None
        self.task = None

    async def stop(self, dialog_id=None):
        async with self._lock:
            if dialog_id is None or dialog_id == self.dialog_id:
                await self._stop()

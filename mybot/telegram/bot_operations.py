"""Prevent workspace-dependent actions before activation or during switching."""

from functools import wraps


def active_operation(method):
    @wraps(method)
    async def guarded(self, update, *args, **kwargs):
        if not self.is_owner(update):
            return
        controller = self.runtime_controller
        if controller is None or self.reply_service is None or self.workspace is None or controller.busy:
            await self.callbacks.show(update.effective_message,
                "Диалог подготавливается…" if controller and controller.busy
                else "Сначала откройте диалог через меню «Диалоги».")
            return
        async with controller.lock:
            return await method(self, update, *args, **kwargs)
    return guarded

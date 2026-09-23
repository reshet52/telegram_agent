"""Compact typing toggle for the active dialog."""

from telegram import InlineKeyboardButton

from mybot.telegram.bot_keyboards import home_menu


async def handle_typing(bot, action, message):
    controller = bot.runtime_controller
    if controller is None or bot.workspace is None:
        await bot.callbacks.show(message, "Сначала откройте диалог.")
        return
    typing = controller.typing
    dialog_id = bot.workspace.dialog_id
    if action == "ui:typing_stop" or (
            action == "ui:typing_toggle" and typing.dialog_id == dialog_id):
        await typing.stop(dialog_id)
    elif action == "ui:typing_toggle" or action.startswith("ui:typing_start:"):
        if action.startswith("ui:typing_start:"):
            try:
                if int(action.rsplit(":", 1)[1]) != dialog_id:
                    raise ValueError
            except ValueError:
                await bot.callbacks.show(message, "Диалог изменился.")
                return
        if controller.busy or controller.lock.locked():
            await bot.callbacks.show(message, "Дождитесь завершения текущей операции.")
            return
        async with controller.lock:
            if bot.workspace is None or bot.workspace.dialog_id != dialog_id:
                await bot.callbacks.show(message, "Диалог изменился.")
                return
            try:
                await typing.start(dialog_id)
            except Exception:
                await bot.callbacks.show(message, "Не удалось включить печать. Попробуйте снова.")
                return
    active = typing.dialog_id == dialog_id
    status = ("пауза перед печатью" if active and typing.phase == "thinking" else
              "включена" if active else "выключена")
    await bot.callbacks.show(message, f"⌨️ Печать: {status}.", home_menu([
        [InlineKeyboardButton("Выключить" if active else "Включить",
                              callback_data="ui:typing_toggle")],
    ]))

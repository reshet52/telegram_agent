"""Explicitly label real outgoing messages for later correction learning."""

from uuid import UUID

from telegram import InlineKeyboardButton

from mybot.storage.reply_journal import (
    finalize_draft_feedback, get_outgoing_reply, link_outgoing_to_draft,
    mark_independent_reply, recent_kept_drafts,
)
from mybot.telegram.bot_keyboards import home_menu


class BotFeedback:
    def __init__(self, bot):
        self.bot = bot

    async def show(self, text, rows=None):
        await self.bot.callbacks.show(None, text, home_menu(rows or []))

    def workspace_for(self, dialog_id):
        controller = self.bot.runtime_controller
        workspace = self.bot.workspace
        if (workspace is None or (dialog_id is not None and workspace.dialog_id != dialog_id)
                or controller is None
                or controller.busy or controller.lock.locked()):
            return None
        return workspace

    async def handle(self, action):
        parts = action.split(":")
        if len(parts) < 4:
            await self.show("Кнопка устарела.")
            return
        try:
            if parts[1] in {'fl', 'ff'}:
                message_id, draft_id = int(parts[2]), str(UUID(hex=parts[3]))
                dialog_id = None
            else:
                dialog_id, message_id = int(parts[2]), int(parts[3])
                draft_id = None
        except ValueError:
            await self.show("Кнопка устарела.")
            return
        workspace = self.workspace_for(dialog_id)
        if workspace is None:
            await self.show("Откройте исходный диалог, чтобы разметить ответ.")
            return
        outgoing = get_outgoing_reply(workspace, message_id)
        if outgoing is None or not outgoing['text']:
            await self.show("Исходящее сообщение не найдено в журнале диалога.")
            return
        operation = parts[1]
        if operation == 'fb':
            drafts = recent_kept_drafts(workspace)
            summary = "\n".join(
                f"{number}. Вариант {draft['chosen_variant']}: "
                f"{draft['final_text'].replace(chr(10), ' ')[:60]}"
                for number, draft in enumerate(drafts, 1))
            rows = [[InlineKeyboardButton(
                str(number),
                callback_data=f"ui:fl:{message_id}:{UUID(draft['draft_id']).hex}")]
                for number, draft in enumerate(drafts, 1)]
            rows.append([InlineKeyboardButton(
                "Написал сам", callback_data=f"ui:fi:{dialog_id}:{message_id}")])
            label = ("Выберите оставленный вариант, к которому относится это сообщение. "
                     "Если ответ состоит из нескольких сообщений, привяжите каждое отдельно.")
            if not drafts:
                label = "Оставленных вариантов пока нет. Можно отметить самостоятельный ответ."
            await self.show(label + ("\n\n" + summary if summary else ""), rows)
        elif operation == 'fi':
            try:
                mark_independent_reply(workspace, message_id)
            except ValueError as error:
                await self.show(str(error))
                return
            await self.show("Отмечено как самостоятельный ответ. Это не считается отклонением вариантов.")
        elif operation == 'fl' and len(parts) == 4:
            try:
                count = link_outgoing_to_draft(workspace, message_id, draft_id)
            except ValueError as error:
                await self.show(str(error))
                return
            await self.show(f"Привязано сообщений: {count}. Если ответ продолжается, "
                "отправьте остальные части и привяжите их так же. Когда закончите, завершите запись.",
                [[InlineKeyboardButton("✅ Это весь ответ",
                    callback_data=f"ui:ff:{message_id}:{UUID(draft_id).hex}")]])
        elif operation == 'ff' and len(parts) == 4:
            try:
                result, _, count = finalize_draft_feedback(workspace, draft_id)
            except ValueError as error:
                await self.show(str(error))
                return
            label = ('без изменений' if result == 'accepted_without_edit' else
                     'с изменениями относительно варианта ИИ')
            await self.show(f"Ответ размечен: {label}. Сообщений: {count}.")
        else:
            await self.show("Кнопка устарела.")

"""One editable suggestion card in the private bot; no contact sending."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError

from mybot.storage.reply_journal import (
    cancel_draft, create_draft, get_draft, get_generation,
    mark_draft_kept, reject_generation, unreject_generation, update_draft_text,
)


MAX_REPLY_LENGTH = 3500


def variants_text(variants):
    return "\n\n".join(f"Вариант {number}\n{variant}" for number, variant in enumerate(variants, 1))


def variants_markup(generation_id):
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(str(number), callback_data=f"ui:rv:{generation_id}:{number}")
         for number in range(1, 4)],
        [InlineKeyboardButton("🚫 Все не подходят", callback_data=f"ui:reject:{generation_id}")],
    ])


class BotReplyActions:
    def __init__(self, bot):
        self.bot = bot

    async def show(self, source, text, markup=None):
        if source is not None:
            await source.edit_text(text, reply_markup=markup)
        else:
            await self.bot.application.bot.send_message(
                chat_id=self.bot.owner_id, text=text, reply_markup=markup)

    def workspace(self):
        controller = self.bot.runtime_controller
        if (controller is None or controller.busy or controller.lock.locked()
                or self.bot.workspace is None):
            return None
        return self.bot.workspace

    async def show_options(self, source, workspace, generation_id):
        generation = get_generation(workspace, generation_id)
        if generation is None:
            await self.show(source, "Варианты этого диалога недоступны.")
            return
        await self.show(source, variants_text([item["text"] for item in generation["variants"]]),
                        variants_markup(generation_id))

    async def show_draft(self, source, workspace, draft_id):
        draft = get_draft(workspace, draft_id)
        if draft is None or draft["status"] != "ready":
            await self.show(source, "Черновик недоступен. Выберите вариант снова.")
            return
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Изменить", callback_data=f"ui:draft:edit:{draft_id}"),
             InlineKeyboardButton("📋 Оставить", callback_data=f"ui:draft:keep:{draft_id}")],
            [InlineKeyboardButton("Отмена", callback_data=f"ui:draft:cancel:{draft_id}")],
        ])
        await self.show(source, f"Вариант {draft['chosen_variant']}\n{draft['final_text']}", markup)

    async def show_edit(self, source, workspace, draft_id, context):
        draft = get_draft(workspace, draft_id)
        if draft is None or draft["status"] != "ready":
            await self.show(source, "Черновик недоступен. Выберите вариант снова.")
            return
        context.user_data["pending_reply_edit"] = (
            workspace.dialog_id, draft_id, source.message_id)
        await self.show(source,
            f"Скопируйте текст, измените его и пришлите мне одним сообщением:\n\n{draft['final_text']}",
            InlineKeyboardMarkup([[
                InlineKeyboardButton("⬅️ Назад", callback_data=f"ui:db:{draft_id}")]]))

    async def handle(self, action, context, source):
        workspace = self.workspace()
        if workspace is None:
            await self.show(source, "Сначала откройте диалог и дождитесь завершения подготовки.")
            return
        if action.startswith("ui:opts:"):
            generation_id = action.split(":", 2)[2]
            unreject_generation(workspace, generation_id)
            await self.show_options(source, workspace, generation_id)
            return
        if action.startswith("ui:db:"):
            await self.show_draft(source, workspace, action.split(":", 2)[2])
            return
        if action.startswith("ui:unreject:"):
            generation_id = action.split(":", 2)[2]
            if not unreject_generation(workspace, generation_id):
                await self.show(source, "Варианты уже недоступны.")
                return
            await self.show_options(source, workspace, generation_id)
            return
        if action.startswith(("ui:rv:", "ui:rve:")):
            try:
                _, _, generation_id, number = action.split(":")
                draft_id = create_draft(workspace, generation_id, int(number))
            except (ValueError, IndexError) as error:
                await self.show(source, str(error) or "Вариант недоступен.")
                return
            if action.startswith("ui:rve:"):
                await self.show_edit(source, workspace, draft_id, context)
            else:
                await self.show_draft(source, workspace, draft_id)
            return
        if action.startswith("ui:reject:"):
            generation_id = action.split(":", 2)[2]
            try:
                reject_generation(workspace, generation_id)
            except ValueError as error:
                await self.show(source, str(error))
                return
            await self.show(source, "Варианты отклонены.", InlineKeyboardMarkup([
                [InlineKeyboardButton("↩️ Отмена", callback_data=f"ui:unreject:{generation_id}")],
            ]))
            return
        if not action.startswith("ui:draft:"):
            await self.show(source, "Откройте варианты заново.")
            return
        try:
            _, _, command, draft_id = action.split(":")
        except ValueError:
            await self.show(source, "Кнопка устарела.")
            return
        draft = get_draft(workspace, draft_id)
        if draft is None:
            await self.show(source, "Черновик относится к другому диалогу или недоступен.")
            return
        if command == "edit":
            await self.show_edit(source, workspace, draft_id, context)
        elif command == "cancel":
            cancel_draft(workspace, draft_id)
            await self.show_options(source, workspace, draft["generation_id"])
        elif command == "keep":
            if draft["status"] != "ready":
                await self.show(source, "Текст уже оставлен или отменён.")
                return
            await self.show(source, f"Оставлен вариант {draft['chosen_variant']}\n{draft['final_text']}")
            mark_draft_kept(workspace, draft_id)
        else:
            # Old send and reconciliation buttons must remain inert.
            await self.show(source, "Отправка собеседнику из бота отключена.")

    async def handle_text(self, text, context):
        pending = context.user_data.pop("pending_reply_edit", None)
        if pending is None:
            return False
        workspace = self.workspace()
        if workspace is None or pending[0] != workspace.dialog_id:
            await self.show(None, "Диалог изменился. Черновик не изменён.")
            return True
        if not text or len(text) > MAX_REPLY_LENGTH:
            context.user_data["pending_reply_edit"] = pending
            await self.show(None, f"Напишите от 1 до {MAX_REPLY_LENGTH} символов одним сообщением.")
            return True
        draft_id = pending[1]
        if not update_draft_text(workspace, draft_id, text):
            await self.show(None, "Черновик недоступен. Выберите вариант снова.")
            return True
        draft = get_draft(workspace, draft_id)
        markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("✏️ Изменить", callback_data=f"ui:draft:edit:{draft_id}"),
             InlineKeyboardButton("📋 Оставить", callback_data=f"ui:draft:keep:{draft_id}")],
            [InlineKeyboardButton("Отмена", callback_data=f"ui:draft:cancel:{draft_id}")],
        ])
        try:
            await self.bot.application.bot.edit_message_text(
                chat_id=self.bot.owner_id, message_id=pending[2],
                text=f"Вариант {draft['chosen_variant']}\n{draft['final_text']}",
                reply_markup=markup)
        except TelegramError:
            await self.show(None, f"Исправленный вариант {draft['chosen_variant']}\n{draft['final_text']}",
                            markup)
        return True

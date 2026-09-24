"""Inline navigation for the private control bot."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup


DIALOGS_PAGE_SIZE = 8


def main_menu(active=False, panel_hidden=False):
    if active:
        rows = [
            [InlineKeyboardButton("✍️ Ответ", callback_data="ui:reply"),
             InlineKeyboardButton("📖 Контекст", callback_data="ui:context")],
            [InlineKeyboardButton("🎭 Настроение", callback_data="ui:mood"),
             InlineKeyboardButton("🧠 Память", callback_data="ui:memory")],
            [InlineKeyboardButton("🔄 Обновить историю и завершить", callback_data="ui:finish_dialog")],
            [InlineKeyboardButton("⬅️ Выйти из диалога", callback_data="ui:leave")],
        ]
    else:
        rows = [
            [InlineKeyboardButton("💬 Диалоги", callback_data="ui:dialogs:0")],
            [InlineKeyboardButton("🖋 Мой стиль", callback_data="ui:style"),
             InlineKeyboardButton("⏳ Подготовка", callback_data="ui:status")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="ui:settings")],
        ]
    rows.append([InlineKeyboardButton(
        "👁 Показать пульт" if panel_hidden else "🫥 Скрыть пульт",
        callback_data="ui:panel_show" if panel_hidden else "ui:panel_hide")])
    return InlineKeyboardMarkup(rows)


def home_menu(extra_rows=None):
    return InlineKeyboardMarkup((extra_rows or []) + [
        [InlineKeyboardButton("🏠 Главное меню", callback_data="ui:home")],
    ])


def dialogs_menu(dialogs, page):
    pages = max(1, (len(dialogs) + DIALOGS_PAGE_SIZE - 1) // DIALOGS_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    start = page * DIALOGS_PAGE_SIZE
    rows = [[InlineKeyboardButton(
        (dialog.name or f"Диалог {dialog.id}")[:60],
        callback_data=f"ui:select:{dialog.id}:{page}",
    )] for dialog in dialogs[start:start + DIALOGS_PAGE_SIZE]]
    navigation = []
    if page > 0:
        navigation.append(InlineKeyboardButton(
            "⬅️ Назад", callback_data=f"ui:dialogs:{page - 1}"))
    if page + 1 < pages:
        navigation.append(InlineKeyboardButton(
            "Вперёд ➡️", callback_data=f"ui:dialogs:{page + 1}"))
    if navigation:
        rows.append(navigation)
    return home_menu(rows), page + 1, pages


PANEL_ACTIONS = {
    "💬 Диалоги": "ui:dialogs:0", "✍️ Ответ": "ui:reply",
    "📖 Контекст": "ui:context",
    "🏠 Меню": "ui:home", "🫥 Скрыть пульт": "ui:panel_hide",
}


def control_panel(active=False):
    labels = (["✍️ Ответ", "📖 Контекст", "🏠 Меню", "🫥 Скрыть пульт"]
              if active else ["💬 Диалоги", "🏠 Меню", "🫥 Скрыть пульт"])
    return ReplyKeyboardMarkup([labels[i:i + 2] for i in range(0, len(labels), 2)],
                               resize_keyboard=True, is_persistent=True,
                               input_field_placeholder="Инструкция для ответа или кнопка пульта")

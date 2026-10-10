"""Inline navigation for the private control bot."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup


DIALOGS_PAGE_SIZE = 8


def detailed_menu(active=False, panel_hidden=False):
    return main_menu(active, panel_hidden)


def main_menu(active=False, panel_hidden=False):
    rows = [[InlineKeyboardButton('Выбор диалога', callback_data='ui:dialogs:0')]]
    if active:
        rows = [[InlineKeyboardButton('Ответ', callback_data='ui:reply'),
                 InlineKeyboardButton('Указания для ответа', callback_data='ui:reply_instruction')],
                [InlineKeyboardButton('Выбор диалога', callback_data='ui:dialogs:0'),
                 InlineKeyboardButton('Обучение', callback_data='ui:learn')]]
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
    'Ответ': 'ui:reply', 'Указания для ответа': 'ui:reply_instruction',
    'Выбор диалога': 'ui:dialogs:0', 'Обучение': 'ui:learn',
    "💬 Выбрать диалог": "ui:dialogs:0",
    "💬 Диалоги": "ui:dialogs:0", "✍️ Ответ": "ui:reply",
    "📝 Указания": "ui:reply_instruction",
    "📖 Контекст": "ui:context", "🔎 История": "ui:history_search",
    "🏠 Меню": "ui:home", "🫥 Скрыть пульт": "ui:panel_hide",
}


def start_panel(active=True):
    return control_panel(active)


def control_panel(active=False):
    rows = ([['Ответ', 'Указания для ответа'], ['Выбор диалога', 'Обучение']]
            if active else [['Выбор диалога']])
    return ReplyKeyboardMarkup(rows, resize_keyboard=True, is_persistent=True,
                               input_field_placeholder='Выберите действие')

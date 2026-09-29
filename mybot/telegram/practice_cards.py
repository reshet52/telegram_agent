"""Bounded exercise cards: one variant or one context page at a time."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from mybot.storage.reply_practice import RATINGS


def button(label, operation, key):
    return InlineKeyboardButton(label, callback_data=f'ui:rp:{operation}:{key}')


def next_button(ws):
    return button('Следующий отрывок →', 'new', f'{ws.account_id}:{ws.dialog_id}')


def transcript(messages):
    return '\n\n'.join(f"{m['sender']}: {m['text']}" for m in messages)


def rating_markup(case, number):
    current = case['ratings'].get(str(number), {}).get('rating')
    buttons = [button(('✓ ' if value == current else '') + label,
                      'rate', f"{case['id']}:{number}:{value}") for value, label in RATINGS.items()]
    return InlineKeyboardMarkup([buttons[:2], buttons[2:]])


def render_case(ws, case, context, expected, *, view='view', number=1, page=0):
    key = case['id']
    rows = []
    number = max(1, min(3, number))
    if view == 'context':
        body = transcript(context)
        title = 'Контекст до скрытого ответа'
    elif view == 'original':
        if not case.get('revealed_at'):
            raise ValueError('Сначала откройте свой ответ.')
        body = transcript(expected)
        title = 'Ваш настоящий ответ'
    elif case['variants']:
        body = case['variants'][number - 1]
        title = f'Вариант {number}/3'
    else:
        body = 'Ваш ответ скрыт. Получите три варианта и оцените их.'
        title = 'Тренировка на истории'
    # 1400 characters also fit Telegram's UTF-16 limit with emoji and controls.
    pages = [body[i:i + 1400] for i in range(0, len(body), 1400)] or ['']
    page = max(0, min(len(pages) - 1, page))
    text = title + (f' · страница {page + 1}/{len(pages)}' if len(pages) > 1 else '')
    if view == 'view':
        incoming = context[-1]['text']
        text += '\n\nСобеседник: ' + incoming[:200] + ('…' if len(incoming) > 200 else '')
    text += '\n\n' + pages[page]
    if len(pages) > 1:
        nav = []
        if page:
            nav.append(button('← Страница', view, f'{key}:{number}:{page - 1}'))
        if page + 1 < len(pages):
            nav.append(button('Страница →', view, f'{key}:{number}:{page + 1}'))
        rows.append(nav)
    if view == 'view' and case['variants']:
        rows += [list(row) for row in rating_markup(case, number).inline_keyboard]
        rows.append([button(('• ' if i == number else '') + str(i), 'view', f'{key}:{i}:0')
                     for i in range(1, 4)])
        rows.append([button('✏️ Мой вариант', 'correct', key),
                     button('Мой реальный ответ', 'reveal', key)])
        rows.append([button('Контекст', 'context', f'{key}:1:0')])
    elif view == 'view':
        rows.append([button('Получить варианты', 'run', key),
                     button('Контекст', 'context', f'{key}:1:0')])
    else:
        rows.append([button('← К вариантам', 'view', f'{key}:{number}:0')])
        if view == 'original':
            rows.append([button('✓ Использовать как пример' if case.get('use_original')
                                else 'Учиться на этом ответе', 'learn', key)])
    active = (not case.get('learning_disabled') and
              (case.get('use_original') or case.get('owner_correction') or
               any(value['rating'] == 'good' for value in case['ratings'].values())))
    if active:
        text += '\n\n✓ Есть пример для похожих ситуаций этого диалога.'
        rows.append([button('Не использовать как пример', 'unlearn', key)])
    rows.append([next_button(ws), InlineKeyboardButton('Меню', callback_data='ui:practice')])
    return text, InlineKeyboardMarkup(rows)

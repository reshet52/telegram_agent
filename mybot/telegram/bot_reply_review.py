"""Explicit owner ratings, distinct from actual sent-message feedback."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.storage.atomic import durable_call
from mybot.storage.reply_journal import REVIEW_LABELS, review_counts, review_generation


def review_markup(generation_id):
    buttons = [InlineKeyboardButton(label, callback_data=f'ui:rr:{generation_id}:{rating}')
               for rating, label in REVIEW_LABELS.items()]
    return InlineKeyboardMarkup([buttons[:2], buttons[2:]])


async def handle_review(bot, action, update):
    if not bot.is_owner(update):
        return
    try:
        ws = bot.goal.workspace()
        async with bot.runtime_controller.lock:
            if action != 'ui:quality':
                parts = action.split(':')
                if len(parts) != 4:
                    raise ValueError('Кнопка устарела.')
                await durable_call(review_generation, ws, parts[2], parts[3])
            counts = review_counts(ws)
            total = sum(counts.values())
            text = f'Оценки ответов · {ws.dialog_name}\n\nОценено: {total}\n'
            text += '\n'.join(f'{label}: {counts.get(key, 0)}' for key, label in REVIEW_LABELS.items())
            text += ('\n\nОценку можно изменить повторным нажатием. Она не означает, что ответ '
                     'отправлен. Реально отправленные сообщения размечаются отдельно.')
            await bot.callbacks.show(None, text, InlineKeyboardMarkup([
                [InlineKeyboardButton('⬅️ Цель диалога', callback_data='ui:goal')]]))
    except ValueError as error:
        await bot.callbacks.show(None, str(error))
    except Exception:
        logging.exception('Не удалось сохранить оценку решения')
        await bot.callbacks.show(None, 'Оценка не сохранена. Попробуйте снова.')

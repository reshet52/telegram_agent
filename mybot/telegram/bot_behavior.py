"""Per-dialog behavior controls, opened from the goal screen."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.storage.atomic import durable_call
from mybot.storage.dialog_behavior import (
    CHOICES, DEFAULTS, LABELS, MAX_TEXT_LENGTH, format_behavior,
    load_behavior, save_behavior_field,
)


FIELDS = dict(zip('dpbsc', DEFAULTS))
PROMPTS = {
    'boundaries': 'Укажите, что агенту разрешено и чего нельзя обещать или делать.',
    'stop_condition': 'Укажите, при каком результате или событии нужно остановиться.',
    'current_state': 'Опишите ваши актуальные намерения, настроение или доступность. '
                     'Это состояние сохранится до вашей правки или очистки.',
}


class BotBehavior:
    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def button(ws, label, field, value):
        return InlineKeyboardButton(label,
            callback_data=f'ui:bp:{ws.account_id}:{ws.dialog_id}:{field}:{value}')

    async def overview(self, ws):
        values = load_behavior(ws)
        rows = []
        for code, field in FIELDS.items():
            if field in CHOICES:
                rows.append([self.button(ws, ('✓ ' if values[field] == value else '') + label,
                                         code, str(index))
                             for index, (value, label) in enumerate(CHOICES[field].items())])
            else:
                rows.append([self.button(ws, '✏️ ' + LABELS[field], code, 'edit'),
                             self.button(ws, 'Очистить', code, 'clear')])
        rows.append([InlineKeyboardButton('⬅️ Цель диалога', callback_data='ui:goal')])
        await self.bot.callbacks.show(None,
            f'Поведение · {ws.dialog_name}\n\n{format_behavior(values)}\n\n'
            'Настройки действуют только здесь. Настойчивость — последовательность без '
            'давления после отказа. Состояние обновляйте вручную, когда оно изменится.',
            InlineKeyboardMarkup(rows))

    async def handle(self, action, update, context):
        if not self.bot.is_owner(update):
            return
        context.user_data.pop('awaiting_behavior', None)
        try:
            ws = self.bot.goal.workspace()
            async with self.bot.runtime_controller.lock:
                if action == 'ui:behavior':
                    await self.overview(ws)
                    return
                parts = action.split(':')
                if (len(parts) != 6 or parts[2:4] != [str(ws.account_id), str(ws.dialog_id)]
                        or parts[4] not in FIELDS):
                    raise ValueError('Диалог изменился. Откройте настройки поведения заново.')
                field, operation = FIELDS[parts[4]], parts[5]
                if field in CHOICES:
                    options = list(CHOICES[field])
                    if operation not in {str(i) for i in range(len(options))}:
                        raise ValueError('Кнопка устарела.')
                    await durable_call(save_behavior_field, ws, field, options[int(operation)])
                elif operation == 'clear':
                    await durable_call(save_behavior_field, ws, field, '')
                elif operation == 'edit':
                    context.user_data['awaiting_behavior'] = (ws.account_id, ws.dialog_id, field)
                    await self.bot.callbacks.show(None,
                        PROMPTS[field] + f' До {MAX_TEXT_LENGTH} символов.\n\n'
                        + (load_behavior(ws)[field] or 'Пока не задано.'),
                        InlineKeyboardMarkup([[InlineKeyboardButton('⬅️ Назад',
                                                                    callback_data='ui:behavior')]]))
                    return
                else:
                    raise ValueError('Кнопка устарела.')
                await self.overview(ws)
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
        except Exception:
            logging.exception('Не удалось изменить поведение диалога')
            await self.bot.callbacks.show(None, 'Настройки не сохранены. Откройте их заново.')

    async def handle_text(self, text, update, context):
        pending = context.user_data.get('awaiting_behavior')
        if pending is None:
            return False
        if not self.bot.is_owner(update):
            return True
        try:
            ws = self.bot.goal.workspace()
            if pending[:2] != (ws.account_id, ws.dialog_id):
                context.user_data.pop('awaiting_behavior', None)
                raise ValueError('Диалог изменился. Настройки не сохранены.')
            if not text.strip():
                raise ValueError('Напишите настройку или нажмите «Назад».')
            async with self.bot.runtime_controller.lock:
                await durable_call(save_behavior_field, ws, pending[2], text)
                context.user_data.pop('awaiting_behavior', None)
                await self.overview(ws)
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error), InlineKeyboardMarkup([
                [InlineKeyboardButton('⬅️ Назад', callback_data='ui:behavior')]]))
        except Exception:
            logging.exception('Не удалось сохранить поведение диалога')
            await self.bot.callbacks.show(None, 'Настройки не сохранены. Повторите ввод.')
        return True

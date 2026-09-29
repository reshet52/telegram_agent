"""Set, inspect and clear a persistent free-text goal for the active dialog."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.storage.atomic import durable_call
from mybot.storage.dialog_goal import MAX_GOAL_LENGTH, clear_goal, load_goal, save_goal


class BotGoal:
    def __init__(self, bot):
        self.bot = bot

    def workspace(self):
        controller = self.bot.runtime_controller
        if (self.bot.workspace is None or controller is None or controller.busy
                or controller.lock.locked()):
            raise ValueError('Сначала откройте диалог и дождитесь завершения текущей операции.')
        return self.bot.workspace

    async def overview(self, ws):
        text = load_goal(ws)
        rows = [[InlineKeyboardButton('✏️ Изменить цель' if text else '🎯 Задать цель',
                                       callback_data=f'ui:goal:set:{ws.dialog_id}')]]
        if text:
            rows.append([InlineKeyboardButton('Сбросить цель', callback_data=f'ui:goal:clear:{ws.dialog_id}')])
        rows.append([InlineKeyboardButton('Прямота, настойчивость и границы',
                                          callback_data='ui:behavior')])
        rows.append([InlineKeyboardButton('Оценки ответов', callback_data='ui:quality')])
        rows.append([InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')])
        await self.bot.callbacks.show(None,
            f'🎯 Цель диалога · {ws.dialog_name}\n\n{text or "Цель пока не задана."}\n\n'
            'Здесь можно задать направление разговора, настроение и ограничения. '
            'Цель действует до вашей правки или сброса и сохраняется после перезапуска.',
            InlineKeyboardMarkup(rows))

    async def handle(self, action, update, context, text=None):
        if not self.bot.is_owner(update):
            return
        context.user_data.pop('awaiting_goal', None)
        try:
            ws = self.workspace()
            async with self.bot.runtime_controller.lock:
                if action in {'ui:goal', 'ui:goal:write'}:
                    if text is not None:
                        await durable_call(save_goal, ws, text)
                    await self.overview(ws)
                    return
                parts = action.split(':')
                if len(parts) != 4 or parts[3] != str(ws.dialog_id):
                    raise ValueError('Диалог изменился. Откройте «Цель диалога» заново.')
                if parts[2] == 'clear':
                    await durable_call(clear_goal, ws)
                    await self.overview(ws)
                elif parts[2] == 'set':
                    current = load_goal(ws)
                    context.user_data['awaiting_goal'] = (ws.account_id, ws.dialog_id)
                    await self.bot.callbacks.show(None,
                        f'Напишите цель одним сообщением — до {MAX_GOAL_LENGTH} символов. '
                        'Можно совместить настроение, желаемый результат и ограничения. '
                        'Новый текст полностью заменит прежний; до сохранения прежняя цель действует.\n\n'
                        + ('Текущая цель для копирования:\n' + current if current else
                           'Например: «Общайся тепло и кратко. Постепенно подведи разговор к спокойной ночи». '),
                        InlineKeyboardMarkup([[InlineKeyboardButton('⬅️ Назад', callback_data='ui:goal')]]))
                else:
                    raise ValueError('Кнопка устарела. Откройте «Цель диалога» заново.')
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
        except Exception:
            logging.exception('Не удалось открыть или изменить цель диалога')
            await self.bot.callbacks.show(None, 'Не удалось сохранить или прочитать цель. Попробуйте снова.')

    async def handle_text(self, text, update, context):
        pending = context.user_data.get('awaiting_goal')
        if pending is None:
            return False
        if not self.bot.is_owner(update):
            return True
        try:
            ws = self.workspace()
            if pending != (ws.account_id, ws.dialog_id):
                context.user_data.pop('awaiting_goal', None)
                raise ValueError('Диалог изменился. Цель не сохранена; откройте её в нужном диалоге.')
            async with self.bot.runtime_controller.lock:
                await durable_call(save_goal, ws, text)
                context.user_data.pop('awaiting_goal', None)
                await self.overview(ws)
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error), InlineKeyboardMarkup([
                [InlineKeyboardButton('⬅️ Назад', callback_data='ui:goal')]]))
        except Exception:
            logging.exception('Не удалось сохранить цель диалога')
            await self.bot.callbacks.show(None, 'Цель не сохранена. Повторите ввод или вернитесь в меню.',
                InlineKeyboardMarkup([[InlineKeyboardButton('⬅️ Назад', callback_data='ui:goal')]]))
        return True

"""Owner-controlled, one-case-at-a-time historical reply practice."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest

from mybot.services.reply_practice import generate_practice
from mybot.storage.atomic import durable_call
from mybot.storage.reply_practice import PracticeStore
from mybot.telegram.practice_cards import button, next_button, rating_markup, render_case


class BotReplyPractice:
    def __init__(self, bot):
        self.bot = bot

    async def overview(self, store):
        cases = store.records()
        rows = [[next_button(store.workspace)]]
        rows.append([InlineKeyboardButton('Локальное обучение', callback_data='ui:learning')])
        for case in cases[:5]:
            label = 'Оценить' if case['variants'] else 'Продолжить'
            rows.append([button(f"{label} · {case['created_at'][:16]} · {len(case['ratings'])}/3",
                                'open', case['id'])])
        rows.append([InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')])
        await self.bot.callbacks.show(None,
            'Тренировка на истории\n\n'
            'Оцените варианты или напишите свой. Одобренные примеры помогают '
            'следующим ответам в этом диалоге.\n'
            'Оценки собирают примеры. Кнопка «Локальное обучение» отдельно запускает '
            'обновление весов на Mac. Варианты пока создаёт текущая модель ответов.\n\n'
            f'Отрывков: {len(cases)}. Последние пять:',
            InlineKeyboardMarkup(rows))

    async def show_case(self, store, case, update, *, view='view', number=1, page=0):
        context, expected = store.source(case)
        text, markup = render_case(store.workspace, case, context, expected,
                                   view=view, number=number, page=page)
        # Only edit an exercise card, never an unrelated dialog message/menu.
        source = update.effective_message
        data = getattr(getattr(update, 'callback_query', None), 'data', '') or ''
        markup_before = getattr(source, 'reply_markup', None)
        rows = getattr(markup_before, 'inline_keyboard', ()) or ()
        is_card = any((b.callback_data or '').startswith((
            'ui:rp:run:', 'ui:rp:rate:', 'ui:rp:view:', 'ui:rp:reveal:'))
            for row in rows for b in row)
        if isinstance(data, str) and data.startswith('ui:rp:') and is_card:
            try:
                await source.edit_text(text, reply_markup=markup)
                return
            except BadRequest as error:
                if 'message is not modified' in str(error).lower():
                    return
                if 'message to edit not found' not in str(error).lower():
                    raise
        await self.bot.send_long_text(source, text, reply_markup=markup)

    async def handle(self, action, update, context):
        if not self.bot.is_owner(update):
            return
        try:
            ws = self.bot.goal.workspace()
            async with self.bot.runtime_controller.lock:
                store = PracticeStore(ws)
                context.user_data.pop('practice_correction', None)
                if action == 'ui:practice':
                    await self.overview(store)
                    return
                parts = action.split(':')
                if len(parts) < 4:
                    raise ValueError('Кнопка устарела. Откройте тренировку заново.')
                operation, key = parts[2:4]
                if operation == 'new' and len(parts) == 5:
                    if parts[3:] != [str(ws.account_id), str(ws.dialog_id)]:
                        raise ValueError('Диалог изменился. Откройте тренировку заново.')
                    case = await durable_call(store.create)
                    await self.show_case(store, case, update)
                elif operation in {'open', 'run', 'reveal'} and len(parts) == 4:
                    case = store.get(key)
                    store.source(case)
                    if operation == 'run':
                        await self.bot.callbacks.show(None, 'Готовлю варианты для исторического отрывка…')
                        case = await generate_practice(store, key)
                    elif operation == 'reveal':
                        case = await durable_call(store.reveal, key)
                        await self.show_case(store, case, update, view='original')
                        return
                    await self.show_case(store, case, update)
                elif operation in {'view', 'context', 'original'} and len(parts) == 6:
                    await self.show_case(store, store.get(key), update, view=operation,
                                         number=int(parts[4]), page=int(parts[5]))
                elif operation in {'learn', 'unlearn', 'correct'} and len(parts) == 4:
                    case = store.get(key)
                    store.source(case)
                    if operation == 'correct':
                        if not case['variants']:
                            raise ValueError('Сначала получите варианты.')
                        context.user_data['practice_correction'] = (ws.account_id, ws.dialog_id, key)
                        await self.bot.callbacks.show(None,
                            'Напишите, как вы ответили бы здесь (до 3500 символов). '
                            'Сохраню как пример для похожих ситуаций этого диалога.',
                            InlineKeyboardMarkup([[button('Отмена', 'open', key)]]))
                        return
                    if operation == 'learn':
                        case = await durable_call(store.use_original, key, True)
                    else:
                        case = await durable_call(store.clear_learning, key)
                    await self.show_case(store, case, update)
                elif operation == 'rate' and len(parts) == 6:
                    number = int(parts[4])
                    case = await durable_call(store.rate, key, number, parts[5])
                    await self.show_case(store, case, update, number=number)
                else:
                    raise ValueError('Кнопка устарела. Откройте тренировку заново.')
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
        except Exception:
            logging.exception('Не удалось выполнить тренировку ответа')
            await self.bot.callbacks.show(None, 'Не удалось выполнить действие. Откройте тренировку '
                'и продолжите сохранённый отрывок. Успешные результаты повторно не генерируются.')

    async def handle_text(self, text, update, context):
        pending = context.user_data.get('practice_correction')
        if pending is None:
            return False
        if not self.bot.is_owner(update):
            return True
        try:
            ws = self.bot.goal.workspace()
            if pending[:2] != (ws.account_id, ws.dialog_id):
                context.user_data.pop('practice_correction', None)
                raise ValueError('Диалог изменился. Исправление не сохранено.')
            async with self.bot.runtime_controller.lock:
                store = PracticeStore(ws)
                case = await durable_call(store.save_correction, pending[2], text)
                context.user_data.pop('practice_correction', None)
                await self.show_case(store, case, update)
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
        except Exception:
            logging.exception('Не удалось сохранить исправление тренировки')
            await self.bot.callbacks.show(None, 'Не удалось сохранить исправление. Повторите ввод.')
        return True

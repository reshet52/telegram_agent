"""Owner-controlled, one-case-at-a-time historical reply practice."""

import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.services.reply_practice import generate_practice
from mybot.storage.atomic import durable_call
from mybot.storage.reply_practice import PracticeStore, RATINGS


def button(label, operation, key):
    return InlineKeyboardButton(label, callback_data=f'ui:rp:{operation}:{key}')


def next_button(ws):
    return button('Следующий отрывок', 'new', f'{ws.account_id}:{ws.dialog_id}')


def transcript(messages):
    return '\n\n'.join(f"{m.get('date') or ''} · {m['sender']}\n{m['text']}" for m in messages)


def rating_markup(case, number):
    current = case['ratings'].get(str(number), {}).get('rating')
    buttons = [button(('✓ ' if value == current else '') + label,
                      'rate', f"{case['id']}:{number}:{value}") for value, label in RATINGS.items()]
    return InlineKeyboardMarkup([buttons[:2], buttons[2:]])


class BotReplyPractice:
    def __init__(self, bot):
        self.bot = bot

    async def overview(self, store):
        cases = store.records()
        rows = [[next_button(store.workspace)]]
        rows.append([InlineKeyboardButton('Данные для дообучения', callback_data='ui:learning')])
        for case in cases[:5]:
            label = 'Оценить' if case['variants'] else 'Продолжить'
            rows.append([button(f"{label} · {case['created_at'][:16]} · {len(case['ratings'])}/3",
                                'open', case['id'])])
        rows.append([InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')])
        await self.bot.callbacks.show(None,
            'Тренировка на истории\n\n'
            'Выберу отрывок перед вашим реальным ответом. Модель получит только предшествующую '
            'переписку, без скрытого ответа, сегодняшней памяти и целей. '
            'Сначала оцените варианты, затем откройте свой ответ для сравнения.\n\n'
            'Один запуск — один запрос к настроенной модели ответов; API может быть платным. '
            'Массового запуска нет. Оценки сохраняются, веса модели пока не обучаются.\n\n'
            f'Сохранено отрывков: {len(cases)}. Показаны последние пять.',
            InlineKeyboardMarkup(rows))

    async def show_case(self, store, case, update):
        context, _ = store.source(case)
        await self.bot.send_long_text(update.effective_message,
            'Отрывок для тренировки · ваш следующий ответ скрыт\n\n' + transcript(context))
        if not case['variants']:
            await self.bot.send_long_text(update.effective_message,
                'Получить три варианта модели для этого момента? Один запрос; '
                'текст выше — весь доступный ей контекст.',
                reply_markup=InlineKeyboardMarkup([
                    [button('Получить варианты', 'run', case['id'])],
                    [next_button(store.workspace)],
                    [InlineKeyboardButton('⬅️ Тренировка', callback_data='ui:practice')]]))
            return
        for number, variant in enumerate(case['variants'], 1):
            await self.bot.send_long_text(update.effective_message,
                f'Тренировка · вариант {number}\n\n{variant}',
                reply_markup=rating_markup(case, number))
        await self.bot.send_long_text(update.effective_message,
            'Оцените каждый вариант. Ваш настоящий ответ можно открыть для сравнения; '
            'оценки после открытия будут помечены отдельно.',
            reply_markup=InlineKeyboardMarkup([
                [button('Показать мой ответ', 'reveal', case['id'])],
                [next_button(store.workspace)],
                [InlineKeyboardButton('⬅️ Тренировка', callback_data='ui:practice')]]))

    async def handle(self, action, update, context):
        if not self.bot.is_owner(update):
            return
        try:
            ws = self.bot.goal.workspace()
            async with self.bot.runtime_controller.lock:
                store = PracticeStore(ws)
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
                        _, expected = store.source(case)
                        await self.bot.send_long_text(update.effective_message,
                            'Ваш настоящий ответ в этой ситуации\n\n' + transcript(expected))
                        return
                    await self.show_case(store, case, update)
                elif operation == 'rate' and len(parts) == 6:
                    number = int(parts[4])
                    case = await durable_call(store.rate, key, number, parts[5])
                    await update.effective_message.edit_reply_markup(reply_markup=rating_markup(case, number))
                else:
                    raise ValueError('Кнопка устарела. Откройте тренировку заново.')
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
        except Exception:
            logging.exception('Не удалось выполнить тренировку ответа')
            await self.bot.callbacks.show(None, 'Не удалось выполнить действие. Откройте тренировку '
                'и продолжите сохранённый отрывок. Успешные результаты повторно не генерируются.')

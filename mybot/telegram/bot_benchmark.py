"""Private benchmark collection and manual evaluation through the control bot."""

import asyncio
import logging
from collections import Counter
from uuid import uuid4

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from mybot.config import Config
from mybot.services.history_search import search_history
from mybot.storage.atomic import durable_call
from mybot.storage.search_benchmark import BenchmarkStore


LABELS = {'correct': 'Верно', 'wrong': 'Неверно', 'not_found': 'Не найдено'}
INPUT_KEYS = ('benchmark_draft', 'benchmark_input', 'benchmark_run', 'benchmark_delete')


def clear_benchmark_input(context):
    for key in INPUT_KEYS:
        context.user_data.pop(key, None)


def button(label, operation, workspace, key=''):
    return InlineKeyboardButton(label, callback_data=f'ui:bm:{operation}:{workspace.dialog_id}:{key}')


def details(case):
    lines = [f"Вопрос: {case.get('question') or 'ещё не задан'}",
             f"Ожидаемый ответ: {case['expected_answer']}", 'Источники:']
    for source in case['sources']:
        lines.append(f"{source['sender']} · {source.get('date') or 'дата неизвестна'} · "
                     f"ID {source['message_id']}\n«{source['quote']}»")
    return '\n\n'.join(lines)


class BotBenchmark:
    def __init__(self, bot):
        self.bot = bot

    async def show(self, text, rows=None):
        rows = list(rows or [])
        rows.append([InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')])
        await self.bot.callbacks.show(None, text, InlineKeyboardMarkup(rows))

    def workspace(self):
        controller = self.bot.runtime_controller
        if (self.bot.workspace is None or controller is None or controller.busy
                or controller.lock.locked()):
            raise ValueError('Сначала откройте диалог и дождитесь завершения текущей операции.')
        return self.bot.workspace

    async def handle(self, action, update, context):
        if not self.bot.is_owner(update):
            return
        try:
            ws = self.workspace()
            if action == 'ui:benchmark':
                operation, key = 'list', ''
            else:
                _, _, operation, dialog_id, key = action.split(':')
                if dialog_id != str(ws.dialog_id):
                    raise ValueError('Кнопка относится к другому диалогу. Откройте исходный диалог.')
            async with self.bot.runtime_controller.lock:
                store = BenchmarkStore(ws)
                if operation != 'run':
                    context.user_data.pop('benchmark_run', None)
                if operation != 'delete_yes':
                    context.user_data.pop('benchmark_delete', None)
                if operation not in {'save', 'question', 'answer', 'preview'}:
                    context.user_data.pop('benchmark_input', None)
                    context.user_data.pop('benchmark_draft', None)
                await self._dispatch(operation, key, store, update, context)
        except ValueError as error:
            await self.show(str(error))
        except Exception:
            logging.exception('Не удалось выполнить действие проверки поиска')
            await self.show('Не удалось выполнить действие. Сохранённые записи остаются на диске; откройте раздел заново.')

    async def _dispatch(self, operation, key, store, update, context):
        ws = store.workspace
        if operation in {'list', 'cancel'}:
            clear_benchmark_input(context)
            await self.list_cases(store)
        elif operation in {'add', 'edit'}:
            record = store.fact(key) if operation == 'add' else store.case(key)
            draft = dict(record, case_id=record['id'] if operation == 'edit' else None,
                         account_id=ws.account_id, dialog_id=ws.dialog_id, token=uuid4().hex[:16])
            context.user_data['benchmark_draft'] = draft
            if operation == 'add':
                await self.ask_field(context, ws, 'question')
            else:
                await self.preview(context, ws)
        elif operation in {'save', 'question', 'answer', 'preview'}:
            draft = self.draft(context, ws, key)
            context.user_data.pop('benchmark_input', None)
            if operation == 'save':
                record = await durable_call(store.save_case, draft)
                clear_benchmark_input(context)
                await self.show_case(store, record['id'])
            elif operation == 'preview':
                await self.preview(context, ws)
            else:
                await self.ask_field(context, ws, 'question' if operation == 'question' else 'expected_answer')
        elif operation == 'view':
            await self.show_case(store, key)
        elif operation == 'result':
            await self.show_run(store, store.run(key), update)
        elif operation == 'delete':
            case = store.case(key)
            context.user_data['benchmark_delete'] = (ws.account_id, ws.dialog_id, key, case['revision'])
            await self.show('Удалить проверочный вопрос? История переписки и журнал прошлых запусков сохранятся.\n\n'
                            + case['question'], [[button('Удалить', 'delete_yes', ws, key)],
                                                 [button('Отмена', 'view', ws, key)]])
        elif operation == 'delete_yes':
            case = store.case(key)
            if context.user_data.pop('benchmark_delete', None) != (ws.account_id, ws.dialog_id, key, case['revision']):
                raise ValueError('Подтверждение удаления устарело.')
            await durable_call(store.delete_case, key)
            await self.list_cases(store)
        elif operation == 'check':
            case = store.case(key)
            context.user_data['benchmark_run'] = (ws.account_id, ws.dialog_id, key, case['revision'])
            await self.show('Проверяется один вопрос текущим поиском. Это платные запросы API; '
                            'эталон модели не передаётся.\n\n' + case['question'],
                            [[button('Запустить проверку', 'run', ws, key)],
                             [button('Отмена', 'view', ws, key)]])
        elif operation == 'run':
            case = store.case(key)
            if context.user_data.pop('benchmark_run', None) != (ws.account_id, ws.dialog_id, key, case['revision']):
                raise ValueError('Подтверждение запуска устарело. Нажмите «Проверить поиск» заново.')
            await self.evaluate(store, case, update)
        elif operation.startswith('rate_'):
            rating = {'rate_ok': 'correct', 'rate_bad': 'wrong', 'rate_none': 'not_found'}.get(operation)
            if rating is None:
                raise ValueError('Кнопка устарела.')
            await durable_call(store.rate, key, rating)
            await self.show('Оценка сохранена: ' + LABELS[rating],
                            [[button('К вопросам', 'list', ws)]])
        else:
            raise ValueError('Кнопка устарела. Откройте раздел проверки заново.')

    def draft(self, context, ws, token=None):
        draft = context.user_data.get('benchmark_draft')
        if (not draft or draft['account_id'] != ws.account_id or draft['dialog_id'] != ws.dialog_id
                or (token is not None and draft['token'] != token)):
            raise ValueError('Ввод отменён или устарел. Начните добавление/изменение заново.')
        return draft

    async def ask_field(self, context, ws, field):
        draft = self.draft(context, ws)
        context.user_data['benchmark_input'] = field
        label = 'проверочный вопрос' if field == 'question' else 'правильный ожидаемый ответ'
        await self.show(f'Напишите {label} одним сообщением (до 1000 символов).',
                        [[button('Назад', 'preview', ws, draft['token'])],
                         [button('Отмена', 'cancel', ws)]])

    async def handle_text(self, text, update, context):
        field = context.user_data.get('benchmark_input')
        if field is None:
            return False
        if not self.bot.is_owner(update):
            return True
        try:
            ws = self.workspace()
            async with self.bot.runtime_controller.lock:
                draft = self.draft(context, ws)
                if not 1 <= len(text.strip()) <= 1000:
                    await self.show('Введите от 1 до 1000 символов.',
                                    [[button('Отмена', 'cancel', ws)]])
                    return True
                if field not in {'question', 'expected_answer'}:
                    raise ValueError('Режим ввода устарел.')
                draft[field] = text.strip()
                context.user_data.pop('benchmark_input', None)
                await self.preview(context, ws)
        except ValueError as error:
            clear_benchmark_input(context)
            await self.show(str(error))
        return True

    async def preview(self, context, ws):
        draft = self.draft(context, ws)
        token = draft['token']
        await self.show('Проверьте вопрос, ответ и цитаты перед сохранением.\n\n' + details(draft), [
            [button('Сохранить', 'save', ws, token)],
            [button('Изменить вопрос', 'question', ws, token), button('Изменить ответ', 'answer', ws, token)],
            [button('Отмена', 'cancel', ws)],
        ])

    async def list_cases(self, store):
        cases, latest = store.cases(), store.latest()
        counts = Counter()
        for case in cases:
            run = latest.get(case['id'])
            if run is None:
                label = 'Не проверено'
            elif run['case_snapshot']['revision'] != case['revision']:
                label = 'Изменено после проверки'
            elif run['status'] == 'error':
                label = 'Ошибка выполнения'
            else:
                label = LABELS.get(run['rating'], 'Ожидает оценки')
            counts[label] += 1
        summary = '\n'.join(f'{label}: {count}' for label, count in counts.items())
        text = f'🧪 Проверка поиска\nСохранено вопросов: {len(cases)}. Ориентир — около 20.\n\n{summary}'
        if not cases:
            text += '\nЗапросите случайный факт, проверьте цитаты и нажмите «В проверку».'
        await self.show(text, [[button(f'{i}. {case["question"][:45]}', 'view', store.workspace, case['id'])]
                               for i, case in enumerate(cases, 1)])

    async def show_case(self, store, key):
        case = store.case(key)
        ws = store.workspace
        rows = [
            [button('Проверить поиск', 'check', ws, key)],
            [button('Изменить', 'edit', ws, key), button('Удалить', 'delete', ws, key)],
            [button('К вопросам', 'list', ws)],
        ]
        latest = store.latest().get(key)
        if latest:
            rows.insert(1, [button('Последний результат', 'result', ws, latest['id'])])
        await self.show(details(case), rows)

    async def evaluate(self, store, case, update):
        await asyncio.to_thread(store.validate_sources, case)
        await self.show('Проверяю один вопрос…')
        try:
            answer = await search_history(store.workspace, case['question'], persist_cache=False)
        except Exception as error:
            logging.exception('Ошибка запуска проверки поиска')
            await durable_call(store.save_run, case, None, Config.OPENAI_MODEL, type(error).__name__)
            await self.show('Ошибка выполнения сохранена отдельно. Она не считается неверным ответом поиска.',
                            [[button('К вопросам', 'list', store.workspace)]])
            return
        run = await durable_call(store.save_run, case, answer, Config.OPENAI_MODEL)
        await self.show_run(store, run, update)
        await self.show('Результат сохранён. Оцените его кнопками под сообщением.',
                        [[button('К вопросам', 'list', store.workspace)]])

    async def show_run(self, store, run, update):
        ws = store.workspace
        if run['status'] == 'error':
            await self.show(f"Запуск {run['created_at']} завершился ошибкой {run['error']}. "
                            'Он не оценивается как качество поиска.',
                            [[button('К вопросам', 'list', ws)]])
            return
        markup = InlineKeyboardMarkup([
            [button('Верно', 'rate_ok', ws, run['id']), button('Неверно', 'rate_bad', ws, run['id']),
             button('Не найдено', 'rate_none', ws, run['id'])],
            [button('К вопросам', 'list', ws)],
            [InlineKeyboardButton('🏠 Главное меню', callback_data='ui:home')],
        ])
        await self.bot.send_long_text(update.effective_message,
            f"🧪 Результат проверки · {run['created_at']}\nМодель: {run['model']}\n"
            f"Оценка: {LABELS.get(run['rating'], 'ожидается')}\n\n"
            f"{details(run['case_snapshot'])}\n\nОтвет поиска:\n{run['answer']}\n\nОцените результат.",
            reply_markup=markup)

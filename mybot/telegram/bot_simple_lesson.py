"""Small random lesson UI, sharing the live mixed generator and GPU priority."""

import asyncio
import json
import logging

from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup
from telegram.error import TelegramError

from mybot.services import simple_lessons as lessons
from mybot.services.mixed_replies import MixedReplies
from mybot.services.online_learning import request_started, request_finished
from mybot.services.reply_service import ReplyService, DEFAULT_INSTRUCTION
from mybot.storage.atomic import write_json
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.style_lesson import StyleLesson


class BotSimpleLesson:
    def __init__(self, bot):
        self.bot = bot
        self.screen = None
        self.task = None
        self.watcher = None
        self.source = None
        self.panel = None
        self.shown = set()
        self.last_text = None
        self.render_lock = asyncio.Lock()

    def leave(self):
        self.screen = None
        for task in (self.task, self.watcher):
            if task and not task.done():
                task.cancel()
        ws = self.bot.workspace
        if ws and getattr(ws, 'root', None):
            state = StyleLesson(ws).read('simple_lesson.json')
            if state:
                state.pop('input_case', None)
                state.pop('clarification_case', None)
                StyleLesson(ws).write('simple_lesson.json', state)

    def generator(self, ws, key):
        _, case = lessons.current(ws, key)
        context, _ = PracticeStore(ws).source(case)
        service = ReplyService(context, {}, workspace=ws)
        return MixedReplies(service, validator=lambda: lessons.current(ws, key), case_id=key,
            facts_builder=lambda c, instruction, *args: json.dumps({'instruction': instruction}, ensure_ascii=False))

    async def open(self, source):
        ws = self.bot.workspace
        if not ws:
            await self.bot.callbacks.show(source, 'Сначала выберите диалог.', self.bot.main_menu())
            return
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        state = lessons.start(ws)
        self.source = source
        self.last_text = None
        self.screen = (str(ws.root), state['cases'][state['index']])
        await self.show_case(ws)

    async def chunks(self, text, marker):
        if marker in self.shown:
            return
        for offset in range(0, len(text), 3900):
            await self.bot.application.bot.send_message(chat_id=self.bot.owner_id, text=text[offset:offset + 3900])
        self.shown.add(marker)

    async def retire_panel(self):
        if self.panel is not None:
            try:
                await self.panel.edit_reply_markup(reply_markup=None)
            except TelegramError:
                logging.warning('Не удалось убрать кнопки предыдущей учебной карточки')
        self.panel = None
        self.last_text = None

    async def show_case(self, ws):
        await self.retire_panel()
        state = StyleLesson(ws).read('simple_lesson.json')
        if state['index'] == len(state['cases']):
            self.screen = None
            self.panel = await self.bot.application.bot.send_message(chat_id=self.bot.owner_id,
                text=f"Занятие завершено. Сохранено ответов: {len(state['choices'])}.\n" + lessons.status_text(ws),
                reply_markup=InlineKeyboardMarkup([[Button('Продолжить', callback_data='ui:learn')],
                                      [Button('Продолжить обучение', callback_data=f'ui:learn:resume:{ws.dialog_id}')],
                                      [Button('Главное меню', callback_data='ui:home')]]))
            return
        key = state['cases'][state['index']]
        self.screen = (str(ws.root), key)
        try:
            _, case = lessons.current(ws, key)
        except ValueError as error:
            await self.bot.callbacks.show(self.source, str(error), self.markup(key, []))
            return
        context, _ = PracticeStore(ws).source(case)
        record = case.get('mixed_record')
        if record and record.get('cancelled'):
            record.pop('cancelled')
            record['stale'] = False
            self.generator(ws, key).save(record)
            case['mixed_record'] = record
            write_json(PracticeStore(ws).path(key), case)
        await self.chunks('Учебная ситуация\n\n' + '\n\n'.join(f"{m['sender']}: {m['text']}" for m in context), key + ':context')
        await self.render(ws, key, case.get('mixed_record'))
        if not case.get('mixed_record'):
            self.launch(self.generate(ws, key))
        elif any(s['status'] == 'pending' for s in case['mixed_record']['sides'].values()):
            # A previous process stopped before publishing its result: retry only unfinished providers.
            generator = self.generator(ws, key)
            record = case['mixed_record']
            for side in record['sides'].values():
                if side['status'] == 'pending':
                    side.update(status='failed', error='Генерация прервалась; можно повторить.')
            generator.save(record)
            await self.publish(ws, key, record)
        if not self.watcher or self.watcher.done():
            self.watcher = asyncio.create_task(self.watch(ws))

    def launch(self, operation):
        async def run():
            try:
                await operation
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception('Ошибка учебной генерации')
        self.task = asyncio.create_task(run())

    async def generate(self, ws, key, side=None):
        request_started(ws)
        try:
            generator = self.generator(ws, key)
            async def update(record):
                await self.publish(ws, key, record)
            if side:
                _, case = lessons.current(ws, key)
                await generator.retry(case['mixed_record']['id'], side, update)
            else:
                await generator.start(DEFAULT_INSTRUCTION, update)
        finally:
            request_finished(ws)

    async def publish(self, ws, key, record):
        case = PracticeStore(ws).get(key)
        case['mixed_record'] = record
        write_json(PracticeStore(ws).path(key), case)
        state = StyleLesson(ws).read('simple_lesson.json')
        active = (state and state['index'] < len(state['cases']) and state['cases'][state['index']] == key)
        if active and self.screen == (str(ws.root), key) and self.bot.workspace == ws:
            await self.render(ws, key, record)

    def markup(self, key, variants, sides=None):
        rows = []
        if variants:
            rows.append([Button(str(v['number']), callback_data=f"ui:learn:pick:{key}:{v['number']}") for v in variants])
        rows.append([Button('Свой ответ', callback_data=f'ui:learn:own:{key}')])
        rows.append([Button('Здесь не отвечать', callback_data=f'ui:learn:wait:{key}')])
        for side, state in (sides or {}).items():
            if state['status'] == 'failed' and state.get('request'):
                title = 'подготовку' if side == 'understanding' else ('API' if side == 'api' else 'локальные')
                rows.append([Button('Повторить ' + title,
                                    callback_data=f'ui:learn:retry:{key}:{side}')])
        rows.extend([[Button('Другой отрывок', callback_data=f'ui:learn:skip:{key}')],
                     [Button('Продолжить обучение', callback_data=f'ui:learn:resume:{self.bot.workspace.dialog_id}')],
                     [Button('Главное меню', callback_data='ui:home')]])
        return InlineKeyboardMarkup(rows)

    async def render(self, ws, key, record):
        async with self.render_lock:
            if self.screen != (str(ws.root), key) or self.bot.workspace != ws:
                return
            state, _ = lessons.current(ws, key)
            prep = (record or {}).get('preparation', {})
            variants = [] if not record or record.get('stale') else record['variants']
            if any(key + ':' + str(v['number']) not in self.shown for v in variants):
                await self.retire_panel()
            for v in variants:
                await self.chunks(f"Вариант {v['number']} · {'API' if v['source'] == 'api' else 'Локальная'}\n{v['text']}", key + ':' + str(v['number']))
            text = f"Ситуация {state['index'] + 1}/{len(state['cases'])}.\n" + lessons.status_text(ws)
            text += '\n\n1–2 — локальные, 3–4 — API. Выберите ответ, который отправили бы, или напишите свой. Выбор сохранится для обучения.'
            text += '\nЕсли разговор закончен — «Здесь не отвечать». Это сохранит решение молчать, а не текст ответа.'
            if prep.get('result', {}).get('response_action') == 'wait':
                text += '\nСейчас лучше не отвечать: ' + prep['result'].get('action_reason', 'разговор завершён.')
            if prep.get('status') == 'needs_clarification':
                state['clarification_case'] = key
                StyleLesson(ws).write('simple_lesson.json', state)
                text += '\n\nДля точного ответа: ' + prep['result']['question'] + '\nОтветьте следующим сообщением.'
            elif prep.get('status') == 'pending':
                if state.pop('clarification_case', None):
                    StyleLesson(ws).write('simple_lesson.json', state)
                text += '\nРазбираю ситуацию и ищу сведения из предшествующей истории…'
            elif prep.get('status') == 'failed':
                text += '\nСмысловая подготовка недоступна; использую доступную переписку.'
            for side, value in (record or {}).get('sides', {}).items():
                if value['status'] == 'pending':
                    text += '\n' + ('API' if side == 'api' else 'Локальная модель') + ': готовлю ответы…'
                elif value['status'] == 'failed':
                    text += '\n' + ('API' if side == 'api' else 'Локальная модель') + ': ' + value['error']
                if any(c.get('status') == 'unavailable' for c in value.get('checks', [])):
                    text += '\nПроверка смысла недоступна — проверьте текст самостоятельно.'
            if record and record.get('stale'):
                text += '\nИсточник изменился. Выберите другой отрывок.'
            if text != self.last_text or self.panel is None:
                sides = dict((record or {}).get('sides', {}))
                if prep.get('status') == 'failed':
                    sides['understanding'] = dict(prep, request=True)
                markup = self.markup(key, variants, sides)
                if self.panel is None:
                    self.panel = await self.bot.application.bot.send_message(
                        chat_id=self.bot.owner_id, text=text, reply_markup=markup)
                else:
                    await self.panel.edit_text(text, reply_markup=markup)
                self.source = self.panel
                self.last_text = text

    async def watch(self, ws):
        while self.screen and self.bot.workspace == ws:
            await asyncio.sleep(5)
            key = self.screen[1] if self.screen else None
            if key:
                try:
                    _, case = lessons.current(ws, key)
                    await self.render(ws, key, case.get('mixed_record'))
                except ValueError:
                    return

    async def handle(self, action, source):
        try:
            if action == 'ui:learn':
                await self.open(source)
                return
            if action.startswith('ui:learn:resume:'):
                ws = self.bot.workspace
                if not ws or action.rsplit(':', 1)[1] != str(ws.dialog_id):
                    raise ValueError('Выберите текущий диалог: кнопка относится к другому диалогу.')
                from mybot.services.online_learning import tick
                message = await tick(ws, manual=True)
                await self.bot.application.bot.send_message(chat_id=self.bot.owner_id,
                    text=message or 'Обучение пока недоступно.')
                return
            _, _, operation, key, *extra = action.split(':')
            ws = self.bot.workspace
            if not ws:
                raise ValueError('Сначала выберите диалог.')
            state = StyleLesson(ws).read('simple_lesson.json')
            if operation == 'wait':
                lessons.confirm_wait(ws, key)
            elif operation == 'skip':
                if not state or state['index'] >= len(state['cases']) or state['cases'][state['index']] != key:
                    raise ValueError('Карточка устарела.')
                lessons.advance(ws, state)
            else:
                state, case = lessons.current(ws, key)
                if operation == 'own':
                    state.pop('clarification_case', None)
                    state['input_case'] = key
                    StyleLesson(ws).write('simple_lesson.json', state)
                    await self.bot.callbacks.show(source, 'Напишите или вставьте исправленный ответ (до 3500 символов).',
                        InlineKeyboardMarkup([[Button('Главное меню', callback_data='ui:home')]]))
                    self.screen = None
                    return
                if operation == 'retry':
                    if self.task and not self.task.done():
                        raise ValueError('Генерация ещё идёт.')
                    self.launch(self.generate(ws, key, extra[0]))
                    return
                if operation != 'pick':
                    raise ValueError('Карточка устарела.')
                record = case.get('mixed_record') or {}
                if record.get('stale') or not self.generator(ws, key).current(record):
                    raise ValueError('Источник изменился.')
                variant = next((v for v in record.get('variants', []) if v['number'] == int(extra[0])), None)
                if not variant:
                    raise ValueError('Вариант пока не готов.')
                lessons.confirm(ws, key, variant['text'], record, variant['number'])
            if self.task and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self.source = source
            self.last_text = None
            await self.show_case(ws)
        except (ValueError, IndexError) as error:
            await self.bot.callbacks.show(source, str(error), self.bot.main_menu())

    async def handle_text(self, text, source):
        ws = self.bot.workspace
        state = StyleLesson(ws).read('simple_lesson.json') if ws else None
        if state and state.get('clarification_case') and not state.get('input_case'):
            key = state['clarification_case']
            try:
                generator = self.generator(ws, key)
                _, case = lessons.current(ws, key)
            except ValueError as error:
                await self.bot.callbacks.show(source, str(error), self.bot.main_menu())
                return True
            state.pop('clarification_case')
            StyleLesson(ws).write('simple_lesson.json', state)
            self.source = source
            async def operation():
                request_started(ws)
                try:
                    await generator.clarify(case['mixed_record']['id'], text,
                                            lambda record: self.publish(ws, key, record))
                except ValueError as error:
                    await self.bot.callbacks.show(source, str(error), self.bot.main_menu())
                finally:
                    request_finished(ws)
            self.launch(operation())
            return True
        if not state or not state.get('input_case'):
            return False
        try:
            lessons.confirm(ws, state['input_case'], text, record={'source': 'owner_edit'})
            if self.task and not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
            self.source = source
            self.last_text = None
            await self.show_case(ws)
        except ValueError as error:
            await self.bot.callbacks.show(source, str(error), self.bot.main_menu())
        return True

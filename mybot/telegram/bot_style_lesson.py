"""One persistent card for a historical lesson, training and blind comparison."""

import asyncio
import logging
import httpx

from telegram import InlineKeyboardButton as Button, InlineKeyboardMarkup as Markup
from telegram.error import BadRequest, TimedOut

from mybot.services import training_jobs, style_comparison, style_versions
from mybot.services.reply_practice import generate_practice
from mybot.services.reviewed_dataset import training_examples, action_examples
from mybot.services.history_sampling import PERIODS
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.atomic import read_json
from mybot.ai.local_format import rejection
from mybot.services.conversation_boundaries import crossing_reason, gap_seconds, set_gap_seconds
from mybot.services.decision_task import confirmed_counts
import re

COMPARISON_ACTIONS = {'compare', 'activate', 'vote', 'cmpage', 'cmpcontext', 'flag'}


def button(text, action, key):
    namespace = 's2' if action in COMPARISON_ACTIONS else 'sl'
    return Button(text, callback_data=f'ui:{namespace}:{action}:{key}')


async def screen_request(call, *, editing=False, **kwargs):
    try:
        return await call(**kwargs)
    except TimedOut as error:
        if not editing and not isinstance(error.__cause__, httpx.ConnectTimeout):
            raise
        # Editing the same card is idempotent; an ambiguous send must not be repeated.
        await asyncio.sleep(1)
        return await call(**kwargs)


class BotStyleLesson:
    def __init__(self, bot):
        self.bot = bot
        self.tasks = {}
        self.screen = None

    def scope(self, ws):
        return f'{ws.account_id}:{ws.dialog_id}'

    def leave_screen(self):
        self.screen = None
        ws = self.bot.workspace
        if ws is not None and hasattr(ws, 'root'):
            lesson = StyleLesson(ws)
            state = lesson.current()
            if state and state.pop('input_case', None):
                lesson.write('lesson.json', state)

    async def show(self, ws, text, rows):
        lesson = StyleLesson(ws)
        card = lesson.read('style_card.json') or {}
        api = self.bot.application.bot
        if card.get('message_id'):
            try:
                await screen_request(api.edit_message_text, editing=True, chat_id=self.bot.owner_id, message_id=card['message_id'],
                                            text=text, reply_markup=Markup(rows))
                return
            except BadRequest as error:
                if 'message is not modified' in str(error).lower():
                    return
                if 'message to edit not found' not in str(error).lower() and "can't be edited" not in str(error).lower():
                    raise
        message = await screen_request(api.send_message, chat_id=self.bot.owner_id, text=text, reply_markup=Markup(rows))
        lesson.write('style_card.json', {'message_id': message.message_id})

    async def overview(self, ws):
        self.screen = ('home', str(ws.root))
        state = StyleLesson(ws).current()
        count = len(training_examples(ws)) + len(action_examples(ws))
        progress = f"Занятие: {state['index']}/{len(state['cases'])}.\n" if state else ''
        if state and state['status'] == 'complete':
            added = sum(choice != 'skip' for choice in state['choices'].values())
            progress += f'Занятие завершено. Выбрано примеров за занятие: {added}.\n'
        rows = [[button('Продолжить занятие' if state and state['status'] != 'complete' else 'Начать занятие · 10 отрывков',
                        'start', self.scope(ws))],
                [button('Обучение и версии', 'job', self.scope(ws))],
                [Button('Главное меню', callback_data='ui:home')]]
        rows.insert(0, [button('Сначала обучить на моей переписке', 'bootstrap', self.scope(ws))])
        if state and state['status'] == 'complete' and count >= 8:
            rows.insert(1, [button('Обучить новую версию', 'train', self.scope(ws))])
        await self.show(ws, f'Обучать мой стиль · {ws.dialog_name}\n\n{progress}'
            f'Подтверждённых примеров: {count} — ответы и решения о паузе '
            '(минимум 8 до проверки длины).\n'
            f'Ответы: {style_versions.label(style_versions.selected(ws))}\n\n'
            f'Занятия: {style_versions.practice_label(ws)}\n\n'
            'В свободное время разбирайте старые ситуации. Модель не видит ваш скрытый ответ. '
            'Выбирайте подходящий вариант, свой ответ или пропускайте отрывок. '
            'Весами занимается отдельная кнопка обучения.', rows)

    def launch(self, key, coroutine):
        if key in self.tasks and not self.tasks[key].done():
            coroutine.close()
            return
        task = asyncio.create_task(coroutine)
        self.tasks[key] = task
        def done(future):
            if self.tasks.get(key) is future:
                self.tasks.pop(key, None)
            if not future.cancelled() and future.exception():
                logging.error('Ошибка фоновой карточки обучения', exc_info=future.exception())
        task.add_done_callback(done)

    async def cancel(self, key):
        task = self.tasks.get(key)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def case(self, ws):
        lesson = StyleLesson(ws)
        state = lesson.current()
        if not state or state['status'] != 'active':
            await self.overview(ws)
            return
        key = state['cases'][state['index']]
        case = lesson.store.get(key)
        try:
            context, expected = lesson.store.source(case)
        except ValueError as error:
            await self.show(ws, str(error), [[button('Пропустить', 'skip', key), button('Пауза', 'pause', key)]])
            return
        if case['expected_ids'] and crossing_reason(context, expected, gap_seconds(ws)):
            await self.show(ws, 'Этот старый отрывок пересекает длинную паузу. Прежние ответы '
                'и оценки сохранены, но использовать его для нового обучения нельзя.',
                [[button('Пропустить', 'skip', key), button('Пауза', 'pause', key)]])
            return
        self.screen = ('case', str(ws.root), key)
        if case.get('model_action', {}).get('action') == 'wait' and not case.get('variants') and case.get('owner_action', {}).get('action') != 'reply':
            preview = '\n'.join(f"{m['sender']}: {m['text']}" for m in context[-3:])[-350:]
            evidence = case.get('boundary_evidence')
            note = ('\nПосле этого в истории видна длинная пауза. Это повод проверить решение, '
                    'а не доказательство, что отвечать не нужно.' if evidence else '')
            rows = [[button('Верно — не отвечать сейчас', 'wait', key)],
                    [button('Ответ всё-таки нужен', 'needreply', key)],
                    [button('Контекст', 'context', f'{key}:0'), button('Пропустить', 'skip', key)],
                    [button('Пауза', 'pause', key)]]
            await self.show(ws, f"Отрывок {state['index'] + 1}/{len(state['cases'])} · "
                f"{style_versions.practice_label(ws, case['model_action']['version'])}\n\n{preview}\n\n"
                f"Модель предлагает сейчас не отвечать.{note}", rows)
            return
        if not case['variants']:
            await self.show(ws, f"Отрывок {state['index'] + 1}/{len(state['cases'])}\n\n"
                'Локальная модель готовит варианты…', [[button('Пропустить', 'skip', key), button('Пауза', 'pause', key)]])
            self.launch(key, self.generate_case(ws, key))
            return
        if any(rejection(t, re.findall(r'\w+|[^\w\s]', t), 'stop') for t in case['variants']):
            await self.show(ws, 'В сохранённых вариантах обнаружен сбой генерации. Старые ответы и оценки сохранены, '
                            'но эти варианты не предлагаются для выбора.', [
                [button('Мой вариант', 'correct', key), button('Мой настоящий ответ', 'original', key)],
                [button('Пропустить', 'skip', key), button('Пауза', 'pause', key)]])
            return
        preview = '\n'.join(f"{m['sender']}: {m['text']}" for m in context[-3:])[-350:]
        if case.get('request', {}).get('version') == 'base':
            await self.show(ws, 'Здесь сохранены старые ответы базовой модели. Они остаются в истории, '
                            'но больше не предлагаются в занятиях.', [
                [button('Пропустить', 'skip', key), button('Пауза', 'pause', key)]])
            return
        titles = ('Первый', 'Второй', 'Третий (сохранённый)')
        preview_limit = 650 if len(case['variants']) == 2 else 450
        body = '\n\n'.join(f'{titles[i - 1]}\n{text[:preview_limit]}{"…" if len(text) > preview_limit else ""}'
                           for i, text in enumerate(case['variants'], 1))
        version = case.get('request', {}).get('version', 'base')
        period = PERIODS[case['period']] if 'period' in case else 'Ранее созданный отрывок'
        date = str(case.get('date') or 'Дата не сохранена')[:19]
        seen = '\nЭтот ответ уже использован для первоначального обучения; здесь можно его исправить.' if style_versions.used_in_history(ws, case) else ''
        rows = [[button(titles[i - 1], 'pick', f'{key}:{i}') for i in range(1, len(case['variants']) + 1)],
                [button('Оба плохие' if len(case['variants']) == 2 else 'Все плохие', 'neither', key)],
                [button('Не отвечать сейчас', 'wait', key)],
                [button('Мой вариант', 'correct', key)] +
                ([button('Мой настоящий ответ', 'original', key)] if case['expected_ids'] else []),
                [button('Контекст', 'context', f'{key}:0'), button('Полные варианты', 'full', f'{key}:0')],
                [button('Что означают отметки?', 'helpflags', key)]]
        for issue, title in (('reaction', 'Не та реакция'), ('facts', 'Выдуманы факты'),
                             ('emoji', 'Чужие эмодзи'), ('wording', 'Не мои фразы')):
            rows.append([button(f'{"✓ " if issue in case.get("issues", {}).get(str(i), []) else ""}{i}: {title}',
                                'issue', f'{key}:{i}:{issue}') for i in range(1, len(case['variants']) + 1)])
        rows.append([button('Пропустить', 'skip', key), button('Пауза', 'pause', key)])
        await self.show(ws, f"Отрывок {state['index'] + 1}/{len(state['cases'])} · {style_versions.practice_label(ws, version)}\n"
            f'{period} · {date}{seen}\n\n{preview}\n\n{body}\n\n'
            'Выберите подходящий ответ или «Не отвечать сейчас». Отметки указывают ошибки; '
            'для обучения ответа нужен выбранный вариант или ваше исправление.', rows)

    async def generate_case(self, ws, key):
        try:
            await generate_practice(StyleLesson(ws).store, key, local=True)
            if self.screen == ('case', str(ws.root), key) and self.bot.workspace == ws:
                await self.case(ws)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if self.screen == ('case', str(ws.root), key) and self.bot.workspace == ws:
                await self.show(ws, f'Варианты не готовы. {error}', [
                    [button('Повторить', 'retry', key), button('Ответ всё-таки нужен', 'forcecase', key)],
                    [button('Пропустить', 'skip', key)],
                    [button('Пауза', 'pause', key)]])

    async def job(self, ws):
        self.screen = ('job', str(ws.root))
        job = training_jobs.status(ws)
        text = (f'Обучение · {ws.dialog_name}\nВыбранных ответов: {len(training_examples(ws))}. '
                f'Подтверждённых решений: {len(action_examples(ws))}.\n')
        text += f'Занятия: {style_versions.practice_label(ws)}\nОтветы: {style_versions.label(style_versions.selected(ws))}\n'
        from mybot.services.online_learning import read_state
        online = read_state(ws)
        if online.get('enabled'):
            text += f"\nПостоянное обучение: очередь {len(online['queue'])}/10; использовано {len(online['consumed'])}.\nЭтап: {online.get('phase')}.\n"
            if online.get('run_id'):
                from mybot.storage.atomic import read_json
                active = read_json(ws.root / 'learning/mlx_runs' / online['run_id'] / 'run.json', {})
                text += f"Шаги: {active.get('completed_steps', 0)}/{active.get('steps', '—')}; {active.get('phase', 'подготовка')}.\nПаузы: {active.get('pause_count', 0)}, {active.get('pause_seconds', 0):.0f} сек.\n"
            if online.get('result'):
                text += online['result'] + '\n'

        text += f'Граница между беседами: {gap_seconds(ws) // 3600} ч.\n'
        decisions = confirmed_counts(ws)
        if decisions['reviewed']:
            text += (f"Решение отвечать: проверено {decisions['reviewed']}; "
                     f"лишних ответов {decisions['wrong_reply']}, "
                     f"ошибочных пауз {decisions['wrong_wait']}.\n")
        rows = []
        rows.append([button('Пауза 2 ч', 'gap', f'2:{self.scope(ws)}'),
                     button('Пауза 4 ч', 'gap', f'4:{self.scope(ws)}'),
                     button('Пауза 8 ч', 'gap', f'8:{self.scope(ws)}')])
        audit = StyleLesson(ws).holdout_audit()
        invalid = sum(bool(item['reason']) for item in audit)
        if invalid:
            text += f'\nВ прежней проверочной выборке {invalid} из {len(audit)} ситуаций пересекают границу разговора.'
            rows.append([button('Посмотреть исправленную проверку', 'fixpreview', self.scope(ws))])
        if job:
            labels = {'preparing': 'Подготавливаю примеры и модель', 'training': 'Обучаю',
                      'checking_before': 'Проверяю исходную модель', 'checking_after': 'Проверяю новую модель',
                      'ready': 'Новая версия готова', 'failed': 'Запуск прерван или завершился ошибкой',
                      'preview_ready': 'Набор проверен. Можно запускать обучение',
                      'checking_generation': 'Проверяю ответы базовой модели и адаптера',
                      'needs_check': 'Веса сохранены; требуется автопроверка ответов',
                      'checked': 'Автопроверка завершена',
                      'stopping': 'Останавливаю', 'stopped': 'Остановлено',
                      'paused': 'Прогресс сохранён', 'cooling': 'Пауза для охлаждения',
                      'checking_validation': 'Проверяю контрольную ошибку', 'loading': 'Загружаю модель'}
            text += '\n' + labels.get(job['state'], job['state'])
            if job.get('run', {}).get('completed_steps'):
                text += f" · {job['run']['completed_steps']}/{job['run'].get('steps', 50)} шагов"
            if job.get('run', {}).get('checked_answers'):
                text += f"\nОтветов проверено: {job['run']['checked_answers']}/{job['run']['total_answers']}"
            if job['state'] == 'ready':
                run = job['run']
                text += f"\nОшибка проверки: {run['loss_before']:.3f} → {run['loss_after']:.3f}.\nКачество оценивается сравнением ответов."
                rows.append([button('Сравнить версии', 'compare', job['id'])])
            candidate = job.get('candidate') or job['id']
            if job['state'] in {'needs_check', 'checked', 'ready'}:
                report = read_json(style_versions.quality_path(ws, candidate), {})
                if report.get('counts'):
                    text += (f"\nТехнически корректны: прежняя {report['counts']['baseline']}, "
                             f"адаптер {report['counts']['candidate']} из {report['expected_per_model']}. "
                             'Это не оценка сходства со стилем.')
                    if report.get('expected_actions'):
                        text += (f"\nРешения отвечать/подождать: корректный формат "
                                 f"{report.get('valid_actions', 0)}/{report['expected_actions']}.")
                if style_versions.quality_passed(ws, candidate):
                    text += '\nАвтопроверка пройдена; адаптер доступен для занятий.'
                    if job['state'] == 'checked':
                        rows.append([button('Новое сравнение', 'compare', candidate)])
                else:
                    text += '\nАдаптер не допущен к занятиям. Старые ответы и оценки сохранены.'
                    rows.append([button('Проверить без переобучения', 'check', candidate)])
                    if report.get('status') == 'failed' and not job.get('full'):
                        rows.append([button('Подготовить исправленное обучение', 'repair', candidate)])
            if job['state'] == 'preview_ready':
                summary = job['run']['summary']
                counts = summary['counts']
                periods = '/'.join(map(str, summary['period_counts']))
                dates = ' — '.join(summary['dates']) or 'Даты в источнике отсутствуют'
                low, high = summary['estimated_minutes']
                full_mode = summary.get('mode') == 'full'
                schedule_text = ('Вся история + повтор недавней трети.' if full_mode else 'Один проход.')
                activation_text = ('Новая версия автоматически не включается.' if full_mode else
                                   'Новая версия включится только в занятиях. Текущая карточка сохранится.')
                text += (f"\nОбучение: {counts['train']}, проверка: {counts['valid']}."
                         f"\nИстория: начало/середина/недавние — {periods}."
                         f"\nИсправлений: {summary['reviewed_count']}.\n{dates}"
                         f'\n{schedule_text} Предварительно ~{low}–{high} мин; длительность зависит от длины текстов.'
                         f'\n{activation_text}')
                if summary.get('mode') == 'full':
                    text += ('\nПолный режим: вся пригодная история + повтор последней трети. '
                             'Предел 8 часов, сохранение прогресса и паузы для охлаждения. '
                             'Новая версия автоматически не включается.')
                sources = summary.get('source_counts', {})
                if sources:
                    actions = sources.get('historical_observed_reply_action', 0) + sources.get('owner_confirmed_action', 0)
                    text += f'\nЗадачи обучения: ответы {counts["train"] - actions}, решения отвечать/подождать {actions}.'
                rows.append([button('Обучить на этом наборе', 'confirm', job['dataset_id'])])
            if training_jobs.alive(job):
                rows.append([button('Остановить обучение', 'stop', self.scope(ws))])
            elif job['state'] == 'failed':
                text += '\n' + job.get('run', {}).get('error', 'Проверьте память и наличие локальной модели.')
                text += '\nРабочая версия сохранена. Журнал: learning/mlx_runs/' + job['id'] + '/process.log.'
                if job.get('kind') == 'check':
                    rows.append([button('Повторить автопроверку', 'check', job['candidate'])])
            if job.get('full') and not training_jobs.alive(job):
                run = job.get('run', {})
                root = ws.root / 'learning/mlx_runs' / job['id']
                if (root / 'checkpoint.json').exists() and run.get('completed_steps', 0) < run.get('steps', 0):
                    rows.append([button('Продолжить · до 8 часов', 'resumefull', job['id'])])
        if not job or not training_jobs.alive(job):
            rows.append([button('Вся переписка · ночное обучение', 'full', self.scope(ws))])
            rows.append([button('Сначала обучить на моей переписке', 'bootstrap', self.scope(ws))])
            if style_versions.practice_state(ws).get('bootstrap'):
                rows.append([button('Доработать по моим оценкам', 'train', self.scope(ws))])
        rows += [[button('Обновить состояние', 'job', self.scope(ws))],
                 [button('Базовая локальная', 'base', self.scope(ws)), button('Прежняя модель', 'configured', self.scope(ws))],
                 [button('Откатить версию', 'rollback', self.scope(ws))],
                 [Button('К занятию', callback_data='ui:lesson')]]
        text += ('\n\nСмешанные ответы: 1–2 локальные, 3–4 API.' if online.get('enabled') else
                 '\n\nЛокальные версии отвечают и обучаются без API. Прежняя API-версия доступна отдельно.')
        await self.show(ws, text, rows)
        if online.get('enabled'):
            self.launch('online:' + str(ws.root), self.watch_online(ws))
        elif job and training_jobs.alive(job):
            self.launch('job:' + job['id'], self.watch_job(ws, job['id']))

    async def watch_online(self, ws):
        while self.screen == ('job', str(ws.root)) and self.bot.workspace == ws:
            await asyncio.sleep(5)
            await self.job(ws)

    async def watch_job(self, ws, key):
        previous = None
        while True:
            job = training_jobs.status(ws)
            if not job or job['id'] != key:
                return
            progress = (job['state'], job.get('run', {}).get('completed_steps'), job.get('run', {}).get('checked_answers'))
            if progress != previous and self.screen == ('job', str(ws.root)) and self.bot.workspace == ws:
                await self.job(ws)
            previous = progress
            if not training_jobs.alive(job):
                return
            await asyncio.sleep(5)

    async def comparison(self, ws, version):
        job = training_jobs.status(ws)
        baseline = job['baseline'] if job and job['id'] == version else 'base'
        data = style_comparison.open_comparison(ws, version, baseline)
        pending = [i for i in range(len(data['cases'])) if str(i) not in data['votes']]
        if not pending:
            counts = {k: list(data['votes'].values()).count(k) for k in ('baseline', 'candidate', 'both', 'neither')}
            await self.show(ws, f"Сравнение завершено\nПрежняя локальная: {counts['baseline']}\n"
                f"Новая: {counts['candidate']}\nОбе подходят: {counts['both']}\nОбе плохие: {counts['neither']}\n\n"
                f"Отмеченные ошибки (прежняя / новая):\n{style_comparison.issue_summary(data)}\nНоль означает отсутствие отметок.\n\n"
                'Это ваши оценки на небольшом наборе, не гарантия улучшения всех ответов.', [
                    [button('Использовать новую версию', 'activate', version)],
                    [button('Оставить прежнюю', 'job', self.scope(ws))]])
            return
        index = pending[0]
        self.screen = ('comparison', str(ws.root), version, index)
        pair = data['pairs'].get(str(index), {})
        if len(pair) != 2:
            await self.show(ws, 'Готовлю два ответа для слепого сравнения…', [
                [button('Обновить сравнение', 'compare', version)],
                [button('Назад', 'job', self.scope(ws))]])
            self.launch('compare:' + version, self.generate_pair(ws, version, index))
            return
        await self.comparison_page(ws, version, index, 0)

    async def comparison_page(self, ws, version, index, page):
        data = style_comparison.open_comparison(ws, version)
        pair = data['pairs'][str(index)]
        first, second = [pair[x]['text'] for x in data['orders'][str(index)]]
        context, _ = StyleLesson(ws).store.source(StyleLesson(ws).store.get(data['cases'][index]))
        start = page * 700
        rows = [[button('Первый', 'vote', f'{version}:{index}:first'), button('Второй', 'vote', f'{version}:{index}:second')],
                [button('Оба подходят', 'vote', f'{version}:{index}:both'), button('Оба плохие', 'vote', f'{version}:{index}:neither')]]
        rows.append([button('Контекст', 'cmpcontext', f'{version}:{index}:0')])
        labels = {'reaction': 'Не та реакция', 'facts': 'Выдуманы факты', 'emoji': 'Чужие эмодзи', 'wording': 'Не мои фразы'}
        for issue, title in labels.items():
            row = []
            for position in (0, 1):
                side = data['orders'][str(index)][position]
                marked = issue in data.get('issues', {}).get(str(index), {}).get(side, [])
                row.append(button(f'{"✓ " if marked else ""}{position + 1}: {title}', 'flag', f'{version}:{index}:{position}:{issue}'))
            rows.append(row)
        if page:
            rows.append([button('← Страница', 'cmpage', f'{version}:{index}:{page - 1}')])
        if max(len(first), len(second)) > start + 700:
            rows.append([button('Страница →', 'cmpage', f'{version}:{index}:{page + 1}')])
        await self.show(ws, f'Сравнение {index + 1}/{len(data["cases"])}\nСобеседник: {context[-1]["text"][:200]}\n\n'
            f'Первый\n{first[start:start + 700]}\n\nВторой\n{second[start:start + 700]}\n\n'
            'При желании отметьте ошибки перед выбором: неверная реакция, выдуманные факты, чужие эмодзи или фразы.', rows)

    async def generate_pair(self, ws, version, index):
        try:
            await style_comparison.generate_pair(ws, version, index)
            if self.screen == ('comparison', str(ws.root), version, index) and self.bot.workspace == ws:
                await self.comparison(ws, version)
        except Exception as error:
            if self.screen == ('comparison', str(ws.root), version, index) and self.bot.workspace == ws:
                await self.show(ws, f'Сравнение не готово. {error}', [[button('Повторить', 'compare', version)]])

    async def handle(self, action, update, context):
        if not self.bot.is_owner(update):
            return
        try:
            ws = self.bot.goal.workspace()
        except ValueError as error:
            await self.bot.callbacks.show(None, str(error))
            return
        lesson = StyleLesson(ws)
        try:
            async with self.bot.runtime_controller.lock:
                self.screen = None
                if action == 'ui:lesson':
                    # Open where the user clicked, not in an old card off-screen.
                    message = getattr(update, 'effective_message', None)
                    if getattr(update, 'callback_query', None) and message is not None:
                        lesson.write('style_card.json', {'message_id': message.message_id})
                        menu = getattr(self.bot, 'menu', None)
                        if menu is not None:
                            menu.message = None
                            menu._remember(None)
                    await self.overview(ws)
                    return
                parts = action.split(':')
                operation, key = parts[2:4]
                if operation == 'gap':
                    if ':'.join(parts[4:]) != self.scope(ws):
                        raise ValueError('Диалог изменился. Откройте обучение заново.')
                    set_gap_seconds(ws, int(key) * 3600)
                    await self.job(ws)
                    return
                if operation in {'fixpreview', 'fixholdout'} and ':'.join(parts[3:]) != self.scope(ws):
                    raise ValueError('Диалог изменился. Откройте обучение заново.')
                if operation == 'fixpreview':
                    options = lesson.save_replacement_preview()
                    lines = [f"{index + 1}. {context[-1].get('date', 'Без даты')} · "
                             f"{expected[0].get('date', 'Без даты')} · "
                             f"ID {context[-1]['message_id']} → {expected[0]['message_id']}"
                             for index, (context, expected) in enumerate(options)]
                    await self.show(ws, 'Новая проверка: ' + str(len(options)) + ' независимых ситуаций.\n'
                        'Старые сравнения и адаптеры сохранятся. После закрепления обучите новую '
                        'исходную версию от базовых весов.\n\n' + '\n'.join(lines),
                        [[button('Закрепить эту проверку', 'fixholdout', self.scope(ws))],
                         [button('Назад', 'job', self.scope(ws))]])
                    return
                if operation == 'fixholdout':
                    lesson.replace_holdout()
                    await self.job(ws)
                    return
                if operation in COMPARISON_ACTIONS and parts[1] != 's2':
                    raise ValueError('Это прежнее сравнение. Оно сохранено отдельно; откройте новое через «Обучение и версии».')
                if operation in {'check', 'repair'}:
                    if operation == 'check':
                        job = await training_jobs.check_version(ws, key)
                    else:
                        job = await training_jobs.preview_history(ws, repair_version=key)
                    self.launch('job:' + job['id'], self.watch_job(ws, job['id']))
                    await self.job(ws)
                    return
                if operation == 'confirm':
                    job = await training_jobs.start(ws, dataset_id=key)
                    self.launch('job:' + job['id'], self.watch_job(ws, job['id']))
                    await self.job(ws)
                    return
                if operation == 'resumefull':
                    job = await training_jobs.resume_full(ws, key)
                    self.launch('job:' + job['id'], self.watch_job(ws, job['id']))
                    await self.job(ws)
                    return
                if operation in {'start', 'job', 'train', 'bootstrap', 'full', 'stop', 'base', 'configured', 'rollback'}:
                    if ':'.join(parts[3:]) != self.scope(ws):
                        raise ValueError('Диалог изменился. Откройте занятие заново.')
                    if operation == 'start':
                        version = style_versions.practice_selected(ws)
                        if version == 'base':
                            raise ValueError('Сначала обучите модель на истории или завершите её автопроверку. '
                                             'Базовая модель в занятиях не используется.')
                        style_versions.adapter(ws, version)
                        lesson.start()
                        await self.case(ws)
                    else:
                        if operation in {'train', 'bootstrap', 'full'}:
                            purpose = 'refine' if operation == 'train' and style_versions.practice_state(ws).get('bootstrap') else 'history'
                            job = await training_jobs.preview_history(ws, purpose, full=operation == 'full')
                            self.launch('job:' + job['id'], self.watch_job(ws, job['id']))
                        elif operation == 'stop':
                            training_jobs.stop(ws)
                        elif operation in {'base', 'configured'}:
                            style_versions.activate(ws, operation)
                        elif operation == 'rollback':
                            style_versions.rollback(ws)
                        await self.job(ws)
                    return
                if operation in COMPARISON_ACTIONS:
                    if operation == 'activate':
                        style_versions.activate(ws, key)
                        await self.overview(ws)
                    else:
                        if operation == 'flag':
                            style_comparison.flag(ws, key, int(parts[4]), int(parts[5]), parts[6])
                            await self.comparison_page(ws, key, int(parts[4]), 0)
                            return
                        if operation == 'vote':
                            style_comparison.vote(ws, key, int(parts[4]), parts[5])
                        if operation == 'cmpcontext':
                            data = style_comparison.open_comparison(ws, key)
                            index, page = int(parts[4]), max(0, int(parts[5]))
                            before, _ = lesson.store.source(lesson.store.get(data['cases'][index]))
                            body = '\n\n'.join(f"{m['sender']}: {m['text']}" for m in before)
                            rows = [[button('К сравнению', 'compare', key)]]
                            if page:
                                rows.append([button('← Страница', 'cmpcontext', f'{key}:{index}:{page - 1}')])
                            if len(body) > (page + 1) * 1400:
                                rows.append([button('Страница →', 'cmpcontext', f'{key}:{index}:{page + 1}')])
                            await self.show(ws, body[page * 1400:(page + 1) * 1400] or 'Пусто', rows)
                        elif operation == 'cmpage':
                            await self.comparison_page(ws, key, int(parts[4]), max(0, int(parts[5])))
                        else:
                            await self.comparison(ws, key)
                    return
                state = lesson.require(key, validate=operation not in {'skip', 'pause'})
                if operation != 'correct' and state.pop('input_case', None):
                    lesson.write('lesson.json', state)
                case = lesson.store.get(key)
                if operation == 'issue':
                    lesson.store.flag_issue(key, int(parts[4]), parts[5])
                    await self.case(ws)
                elif operation == 'helpflags':
                    await self.show(ws, 'Отмечайте ошибки отдельно у каждого ответа; можно несколько, повторное нажатие снимает отметку.\n\n'
                        'Не та реакция — ответ не подходит этой ситуации.\n'
                        'Выдуманы факты — добавлены неизвестные обстоятельства.\n'
                        'Чужие эмодзи — не ваши или неуместные смайлики.\n'
                        'Не мои фразы — формулировка не в вашей манере; новое слово само по себе не ошибка.\n\n'
                        'Отметки не меняют веса и не объясняют модели правильный ответ. Вариант с отметками '
                        'не используется как положительный пример. Напишите исправление или выберите другой ответ. '
                        'Выборы сохраняются для отдельного запуска доработки.', [[button('К вариантам', 'retry', key)]])
                elif operation == 'neither':
                    lesson.store.clear_learning(key)
                    await self.show(ws, 'Эти варианты не добавлены в обучение. Покажите подходящий ответ или пропустите отрывок.', [
                        [button('Мой вариант', 'correct', key), button('Мой настоящий ответ', 'original', key)],
                        [button('Пропустить', 'skip', key), button('К вариантам', 'retry', key)]])
                elif operation == 'pause':
                    lesson.pause()
                    await self.cancel(key)
                    await self.overview(ws)
                elif operation in {'needreply', 'forcecase'}:
                    lesson.store.confirm_action(key, 'reply')
                    await self.case(ws)
                elif operation in {'pick', 'use', 'skip', 'wait'}:
                    if operation == 'skip':
                        await self.cancel(key)
                    elif operation == 'wait':
                        lesson.store.clear_learning(key)
                        lesson.store.confirm_action(key, 'wait')
                    if operation == 'pick':
                        if case.get('issues', {}).get(parts[4]):
                            raise ValueError('У этого ответа отмечены ошибки. Напишите исправленный вариант '
                                             'или снимите ошибочные отметки перед выбором.')
                        lesson.store.rate(key, int(parts[4]), 'good')
                        lesson.store.confirm_action(key, 'reply')
                    elif operation == 'use':
                        lesson.store.use_original(key, True)
                        lesson.store.confirm_action(key, 'reply')
                    lesson.advance(key, operation)
                    await self.case(ws)
                elif operation == 'correct':
                    state = lesson.current()
                    state['input_case'] = key
                    lesson.write('lesson.json', state)
                    await self.show(ws, 'Напишите свой ответ одним сообщением (до 3500 символов).',
                                    [[button('Назад', 'retry', key)]])
                elif operation in {'context', 'full', 'original'}:
                    before, expected = lesson.store.source(case)
                    if operation == 'original':
                        lesson.store.reveal(key)
                    body = ('\n\n'.join(case['variants']) if operation == 'full' else
                            '\n\n'.join(f"{m['sender']}: {m['text']}" for m in (expected if operation == 'original' else before)))
                    page = int(parts[4]) if len(parts) > 4 else 0
                    rows = [[button('К вариантам', 'retry', key)]]
                    if operation == 'original':
                        rows.insert(0, [button('Использовать мой ответ', 'use', key)])
                    if page:
                        rows.append([button('← Страница', operation, f'{key}:{page - 1}')])
                    if len(body) > (page + 1) * 1400:
                        rows.append([button('Страница →', operation, f'{key}:{page + 1}')])
                    await self.show(ws, body[page * 1400:(page + 1) * 1400] or 'Пусто', rows)
                elif operation == 'retry':
                    state = lesson.current()
                    state.pop('input_case', None)
                    lesson.write('lesson.json', state)
                    await self.case(ws)
                else:
                    raise ValueError('Неизвестная кнопка занятия.')
        except (ValueError, IndexError) as error:
            await self.show(ws, str(error), [[Button('К занятию', callback_data='ui:lesson')]])
        except Exception:
            logging.exception('Ошибка занятия')
            await self.show(ws, 'Действие не завершено. Прогресс сохранён; откройте занятие повторно.',
                            [[Button('К занятию', callback_data='ui:lesson')]])

    async def handle_text(self, text, update):
        if not self.bot.is_owner(update) or self.bot.workspace is None:
            return False
        ws = self.bot.workspace
        lesson = StyleLesson(ws)
        state = lesson.current()
        if not state or not state.get('input_case') or state['status'] != 'active':
            return False
        try:
            async with self.bot.runtime_controller.lock:
                key = state['input_case']
                lesson.require(key)
                lesson.store.save_correction(key, text)
                lesson.store.confirm_action(key, 'reply')
                state.pop('input_case')
                lesson.write('lesson.json', state)
                lesson.advance(key, 'correction')
                await self.case(ws)
        except ValueError as error:
            await self.show(ws, str(error), [[Button('К занятию', callback_data='ui:lesson')]])
        return True

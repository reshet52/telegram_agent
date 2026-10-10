"""Random reply-only lessons and durable, explicitly confirmed training evidence."""

import random
from uuid import uuid4

from mybot.services.conversation_boundaries import gap_seconds, wait_review_candidates
from mybot.services.online_dataset import reserved_ids
from mybot.services.online_learning import enable, queue_lock, read_state, state_path
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, candidates, digest, now
from mybot.storage.style_lesson import StyleLesson


def start(ws):
    lesson = StyleLesson(ws)
    state = lesson.read('simple_lesson.json')
    if state and state['index'] < len(state['cases']):
        return state
    reserved = reserved_ids(ws)
    store = PracticeStore(ws)
    used = {c['expected_ids'][0] for c in store.records()
            if c.get('purpose') == 'simple_lesson' and c['expected_ids']}
    options = [(c, e) for c, e in candidates(store.history(),
        load_deleted_message_ids(ws.deleted_message_ids), used, threshold=gap_seconds(ws))
        if not reserved.intersection(m['message_id'] for m in c + e)]
    used_contexts = {tuple(c['context_ids']) for c in store.records() if c.get('purpose') == 'simple_lesson'}
    endings = [(c, []) for c, _ in wait_review_candidates(store.history(), load_deleted_message_ids(ws.deleted_message_ids),
        threshold=gap_seconds(ws)) if tuple(m['message_id'] for m in c) not in used_contexts
        and not reserved.intersection(m['message_id'] for m in c)]
    if not options and not endings:
        raise ValueError('Подходящих новых текстовых ситуаций пока нет.')
    # Include observed session endings for owner review, never label silence automatically.
    chosen = random.sample(endings, min(2, len(endings)))
    chosen += random.sample(options, min(10 - len(chosen), len(options)))
    random.shuffle(chosen)
    state = {'id': uuid4().hex, 'cases': [store.create_from(c, e, 'simple_lesson')['id'] for c, e in chosen],
             'index': 0, 'choices': {}, 'created_at': now()}
    lesson.write('simple_lesson.json', state)
    enable(ws)
    return state


def current(ws, key):
    state = StyleLesson(ws).read('simple_lesson.json')
    if not state or state['index'] >= len(state['cases']) or state['cases'][state['index']] != key:
        raise ValueError('Карточка устарела. Откройте «Обучение».')
    store = PracticeStore(ws)
    case = store.get(key)
    store.source(case)
    if reserved_ids(ws).intersection(case['context_ids'] + case['expected_ids']):
        raise ValueError('Эта ситуация предназначена для проверки, а не обучения.')
    return state, case


def confirmed(ws, item):
    evidence = StyleLesson(ws).read('confirmed/' + item['review_id'] + '.json')
    if not evidence or evidence['event_id'] != item['id'] or evidence['text'] != item['text']:
        raise ValueError('Подтверждённый учебный ответ изменился.')
    store = PracticeStore(ws)
    case = store.get(evidence['case_id'])
    store.source(case)
    if case['source_digest'] != item.get('original_source_digest', item['source_digest']):
        raise ValueError('Источники учебного ответа изменились.')
    return evidence


def confirm(ws, key, text, record=None, number=None):
    state, case = current(ws, key)
    if not isinstance(text, str) or not text.strip() or len(text) > 3500:
        raise ValueError('Напишите ответ длиной от 1 до 3500 символов.')
    text = text.strip()
    # One evidence identity for an unchanged situation/target, even across restarts.
    review_key = digest([case['context_ids'], case['expected_ids']])[:32]
    event_id = digest([review_key, text])
    review_id = event_id[:32]
    evidence = {'case_id': key, 'event_id': event_id, 'text': text,
                'record': record, 'number': number, 'confirmed_at': now(), 'context_ids': case['context_ids'], 'response_ids': case['expected_ids']}
    enable(ws)
    with queue_lock(ws):
        queue = read_state(ws)
        prior = next((q for q in queue['queue'] if q.get('review_key') == review_key), None)
        existing = StyleLesson(ws).read('confirmed/' + prior['review_id'] + '.json') if prior else None
        active_run = queue.get('run_id')
        run = read_json(ws.root / 'learning/mlx_runs' / active_run / 'run.json', {}) if active_run else {}
        frozen = read_json(ws.root / 'learning/reviewed' / (run.get('dataset_id', '') + '.json'), {}).get('new_events', [])
        if existing and existing['event_id'] != event_id and existing['event_id'] in frozen and run.get('status') in {'training', 'paused'}:
            raise ValueError('Предыдущий ответ сейчас участвует в обучении. Дождитесь завершения цикла.')
        StyleLesson(ws).write('confirmed/' + review_id + '.json', evidence)
        queue['queue'] = [q for q in queue['queue'] if q.get('review_key', q.get('review_id')) != review_key]
        if event_id not in queue['consumed']:
            queue['queue'].append({'id': event_id, 'review_id': review_id, 'review_key': review_key, 'source_kind': 'owner_confirmed_reply',
                'context_ids': case['context_ids'], 'response_ids': case['expected_ids'],
                'source_digest': case['source_digest'], 'text': text,
                'provenance': {'generation_id': (record or {}).get('id'), 'chosen_variant': number}})
        write_json(state_path(ws), queue)
    state['choices'][key] = event_id
    advance(ws, state)


def advance(ws, state):
    state.pop('input_case', None)
    state.pop('clarification_case', None)
    state['index'] += 1
    StyleLesson(ws).write('simple_lesson.json', state)


def confirm_wait(ws, key):
    state, case = current(ws, key)
    review_key = digest([case['context_ids'], case['expected_ids']])[:32]
    with queue_lock(ws):
        queue = read_state(ws)
        run = read_json(ws.root / 'learning/mlx_runs' / queue.get('run_id', '') / 'run.json', {})
        frozen = read_json(ws.root / 'learning/reviewed' / (run.get('dataset_id', '') + '.json'), {}).get('new_events', [])
        matching = [q for q in queue.get('queue', []) if q.get('review_key') == review_key]
        if run.get('status') in {'training', 'paused'} and any(q['id'] in frozen for q in matching):
            raise ValueError('Этот пример уже участвует в текущем цикле. Дождитесь завершения.')
        queue['queue'] = [q for q in queue.get('queue', []) if q.get('review_key') != review_key]
        write_json(state_path(ws), queue)
        PracticeStore(ws).confirm_action(key, 'wait')
    state['choices'][key] = 'wait'
    advance(ws, state)


def status_text(ws):
    state = read_state(ws)
    text = f"В очереди обучения: {len(state.get('queue', []))} примеров."
    comparison = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
    if comparison.get('status') == 'running' and (comparison.get('account_id'), comparison.get('dialog_id')) == (ws.account_id, ws.dialog_id):
        text += (' Сравнение ответов: охлаждение Mac.' if comparison.get('phase') == 'cooling' else
                 f" Проверяю скорость локальных ответов: {comparison.get('completed_modes', 0)}/{comparison.get('total_modes', 10)}." if comparison.get('kind') == 'worker_benchmark' else
                 f" Проверяю локальные версии: {comparison.get('completed_modes', 0)}/{comparison.get('total_modes', 30)} прогонов." if comparison.get('kind') == 'local_diagnostics' else
                 f" Сравниваю ответы: {comparison.get('completed_modes', 0) // 2}/{comparison.get('total_modes', 100) // 2}.")
    if state.get('run_id'):
        run = read_json(ws.root / 'learning/mlx_runs' / state['run_id'] / 'run.json', {})
        if run.get('status') == 'training':
            phase = run.get('phase')
            text += (' Охлаждение Mac.' if phase == 'cooling' else
                     f" Обучаю: {run.get('completed_steps', 0)}/{run.get('steps', '…')} шагов.")
        elif run.get('status') == 'paused':
            text += ' Обучение сохранено и приостановлено.'
    if state.get('result'):
        text += '\n' + ('Обучение было остановлено по лимиту времени или ради ответа; текущая модель сохранена.'
                        if state['result'] == 'Обучение не применено: BudgetExpired' else state['result'])
    from mybot.services.style_versions import selected
    if selected(ws) in {'base', 'configured'}:
        text += '\nЛокальная модель этого диалога ещё не обучена. Ваши оценки сохраняются.'
    return text

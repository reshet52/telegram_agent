"""Durable actual-sent reply queue and idle worker scheduling, isolated by dialog."""

import asyncio
import os
import time
from contextlib import contextmanager, closing
from datetime import datetime
from uuid import uuid4

from mybot.services.conversation_boundaries import crossing_reason, gap_seconds
from mybot.services.local_training import PROJECT, PYTHON, MODEL
from mybot.services.online_dataset import reserved_ids
from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, digest
from mybot.storage.style_lesson import StyleLesson

_processes = {}
_active = {}


@contextmanager
def queue_lock(ws):
    import fcntl
    path = ws.root / 'learning/online_queue.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def state_path(ws):
    return ws.root / 'learning/online_learning.json'


def read_state(ws):
    value = read_json(state_path(ws), {})
    if value and (value.get('account_id'), value.get('dialog_id')) != (ws.account_id, ws.dialog_id):
        raise ValueError('Очередь принадлежит другому диалогу.')
    return value


def enable(ws, *, clock=time.time):
    if read_state(ws):
        return
    history = PracticeStore(ws).history()
    write_json(state_path(ws), {'enabled': True, 'account_id': ws.account_id, 'dialog_id': ws.dialog_id,
        'after_id': max((m['message_id'] for m in history), default=0),
        'queue': [], 'consumed': [], 'excluded': [], 'phase': 'collecting', 'enabled_at': clock()})


def request_started(ws):
    if ws is None:
        return
    key = str(ws.root)
    _active[key] = _active.get(key, 0) + 1
    write_json(ws.root / 'learning/online_request.json', {'at': time.time(), 'active': _active[key], 'pid': os.getpid()})
    comparison = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
    if comparison.get('status') == 'running' and comparison.get('pid') != os.getpid():
        write_json(ws.root / 'reply_assistant/ui_request.json', {'at': time.time(), 'pid': os.getpid()})
    write_json(ws.root / 'learning/online_pause.json', {'pause': True})
    for other_root, process in _processes.items():
        if process.returncode is None:
            from pathlib import Path
            write_json(Path(other_root) / 'learning/online_pause.json', {'pause': True})


def request_finished(ws):
    if ws is None:
        return
    key = str(ws.root)
    _active[key] = max(0, _active.get(key, 1) - 1)
    write_json(ws.root / 'learning/online_request.json', {'at': time.time(), 'active': _active[key], 'pid': os.getpid()})


async def pause_for_reply(ws):
    """Wait for the worker's OS lock, never kill a partially saved step."""
    if ws is None:
        return
    write_json(ws.root / 'learning/online_pause.json', {'pause': True})
    state = read_state(ws)
    others_running = any(p.returncode is None for p in _processes.values())
    comparison = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
    comparing = comparison.get('status') == 'running' and comparison.get('pid') != os.getpid()
    if not state.get('run_id') and not others_running and not comparing:
        return
    root = ws.root / 'learning/mlx_runs' / state['run_id'] if state.get('run_id') else None
    if root and read_json(root / 'run.json', {}).get('status') != 'training' and not others_running and not comparing:
        return
    from mybot.ai.local_worker import worker
    await worker.release()
    started = time.monotonic()
    import fcntl
    while True:
        with (PROJECT / 'local_models/.training.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
        if time.monotonic() - started > 180:
            raise ValueError('Обучение ещё сохраняет шаг. API-варианты доступны; повторите локальную часть позже.')
        await asyncio.sleep(.5)


def timestamp(message):
    return datetime.fromisoformat(message['date'].replace('Z', '+00:00')).timestamp()


def collect(ws, *, clock=time.time):
    with queue_lock(ws):
        return _collect(ws, clock=clock)


def _collect(ws, *, clock=time.time):
    state = read_state(ws)
    if not state.get('enabled'):
        return state
    history = PracticeStore(ws).history()
    deleted = load_deleted_message_ids(ws.deleted_message_ids)
    reserved = reserved_ids(ws)
    # Earlier versions excluded a whole turn for any media in its 24-message context.
    # Reconsider those events with a valid text suffix; keep exclusions auditable.
    reconsider = {q['id'] for q in state['excluded'] if q['reason'] == 'nontext_source'}
    used = set(state['consumed']) | {q['id'] for q in state['queue']} | ({q['id'] for q in state['excluded']} - reconsider)
    # Silence closes a turn; later owner messages form a separate event even after restart.
    groups, group = [], []
    for index, message in enumerate(history):
        if (message['sender'] == 'Я' and message['message_id'] > state['after_id']
                and timestamp(message) >= state['enabled_at']):
            if group and timestamp(message) - timestamp(group[-1][1]) >= 30:
                groups.append(group)
                group = []
            group.append((index, message))
        elif group:
            groups.append(group)
            group = []
    if group and clock() - timestamp(group[-1][1]) >= 30:
        groups.append(group)
    for grouped in groups:
        response = [m for _, m in grouped]
        event_id = digest([m['message_id'] for m in response])
        if event_id in used:
            continue
        start = grouped[0][0]
        context = history[max(0, start - 24):start]
        for pos in range(len(context) - 1, -1, -1):
            m = context[pos]
            if (m.get('type', 'text') != 'text' or not isinstance(m.get('text'), str) or not m['text'].strip()):
                context = context[pos + 1:]
                break
        ids = [m['message_id'] for m in context + response]
        reason = crossing_reason(context, response, gap_seconds(ws)) if context else 'no_context'
        if context and context[-1]['sender'] == 'Я':
            reason = 'no_new_incoming_turn'
        if set(ids) & (reserved | deleted):
            reason = 'deleted_or_reserved_source'
        if any(m.get('type', 'text') != 'text' or not isinstance(m.get('text'), str)
               or not m['text'].strip() for m in context + response):
            reason = 'nontext_source'
        if reason:
            if event_id not in reconsider:
                state['excluded'].append({'id': event_id, 'response_ids': [m['message_id'] for m in response], 'reason': reason})
            continue
        state['excluded'] = [q for q in state['excluded'] if q['id'] != event_id]
        state['queue'].append({'id': event_id, 'context_ids': [m['message_id'] for m in context],
            'response_ids': [m['message_id'] for m in response], 'source_digest': digest(context + response),
            'text': '\n\n'.join(m['text'] for m in response), 'source_kind': 'actual_sent_reply', 'provenance': provenance(ws, response)})
    # Remove revoked/deleted queued evidence before launch; never mutate the source history.
    keep = []
    for item in state['queue']:
        if set(item['context_ids'] + item['response_ids']) & (deleted | reserved):
            state['excluded'].append({'id': item['id'], 'reason': 'deleted_or_reserved_source'})
        else:
            keep.append(item)
    state['queue'] = keep
    write_json(state_path(ws), state)
    return state


def provenance(ws, response):
    # Only explicit journal links count. Similar text alone never identifies a proposal.
    import sqlite3
    if not ws.reply_journal.exists():
        return None
    with closing(sqlite3.connect(ws.reply_journal)) as db:
        rows = [db.execute('SELECT generation_id, chosen_variant FROM outgoing_replies WHERE message_id=? AND account_id=? AND dialog_id=?',
                           (m['message_id'], ws.account_id, ws.dialog_id)).fetchone() for m in response]
    if not rows or any(not r or not r[0] for r in rows) or len(set(rows)) != 1:
        return None
    return {'generation_id': rows[0][0], 'chosen_variant': rows[0][1]}


async def tick(ws, *, manual=False):
    state = collect(ws)
    if not state.get('enabled'):
        return 'Обучение этого диалога не включено.'
    comparison = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
    if comparison.get('status') == 'running':
        if (comparison.get('account_id'), comparison.get('dialog_id')) != (ws.account_id, ws.dialog_id):
            raise ValueError('Чужая задача проверки в каталоге диалога.')
        try:
            os.kill(comparison['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            return 'Сначала дождитесь завершения сравнения ответов.'
    request = read_json(ws.root / 'learning/online_request.json', {})
    if request.get('active') and time.time() - request.get('at', 0) < 600:
        return 'Сейчас готовятся ответы. Запустите обучение после их завершения.'
    if not manual and time.time() - request.get('at', state['enabled_at']) < 600:
        return
    from mybot.services.style_versions import selected
    if selected(ws) in {'base', 'configured'}:
        return 'Для этого диалога ещё нет обученного локального адаптера.'
    key = str(ws.root)
    process = _processes.get(key)
    if process and process.returncode is None:
        return 'Обучение уже работает.'
    run_id = state.get('run_id')
    run = read_json(ws.root / 'learning/mlx_runs' / run_id / 'run.json', {}) if run_id else {}
    if run.get('status') == 'training':
        # OS-level scheduler lock in the worker makes stale crash recovery safe.
        try:
            os.kill(run['pid'], 0)
            return 'Обучение уже работает.'
        except ProcessLookupError:
            run['status'] = 'paused'
    if run_id and run.get('status') == 'waiting_gpu':
        # This job never reached training; it has no optimizer checkpoint to resume.
        resume = False
    elif run_id and run.get('status') == 'paused':
        if run.get('cycle_elapsed_seconds', 0) >= 1740:
            return 'Этот цикл достиг предела 30 минут; требуется отдельное продолжение с новым лимитом.'
        resume = True
    elif run_id and not run:
        return 'Сведения о сохранённом обучении отсутствуют. Нужна проверка запуска.'
    else:
        if run_id and run.get('status') == 'interrupted_or_failed':
            return 'Предыдущий запуск завершился ошибкой. Нужна проверка перед продолжением.'
        if run_id:
            state.pop('run_id', None)
        if len(state['queue']) < 10:
            write_json(state_path(ws), state)
            return f"Для нового цикла нужно 10 примеров; сейчас в очереди {len(state['queue'])}."
        resume = False
        run_id = uuid4().hex
        state.update(run_id=run_id, phase='preparing')
        write_json(state_path(ws), state)
    write_json(ws.root / 'learning/online_pause.json', {'pause': False})
    from mybot.ai.local_worker import worker
    await worker.release()
    root = ws.root / 'learning/mlx_runs' / run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'training.log').open('ab') as log:
        args = [str(PYTHON), '-m', 'scripts.online_reply_training', '--account', str(ws.account_id),
                '--dialog', str(ws.dialog_id), '--run-id', run_id]
        if resume:
            args.append('--resume')
        _processes[key] = await asyncio.create_subprocess_exec(*args, cwd=PROJECT, stdout=log, stderr=log,
            start_new_session=True, env=dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1'))
    return 'Обучение продолжено.' if resume else 'Обучение запущено.'


async def monitor(controller):
    import logging
    indexed = {}
    while True:
        await asyncio.sleep(5)
        runtime = controller.runtime
        if runtime and not controller.busy and not controller.lock.locked():
            try:
                # Changes to actual history/deletions refresh evidence, without API calls.
                ws = runtime.workspace
                fingerprint = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None
                                    for p in (ws.chat_history, ws.deleted_message_ids))
                from mybot.services.mixed_replies import enabled as mixed_enabled
                if mixed_enabled(ws) and indexed.get(str(ws.root)) != fingerprint:
                    from mybot.services.reply_memory import ReplyMemory
                    from mybot.storage.atomic import durable_call
                    await durable_call(ReplyMemory(ws).sync)
                    from mybot.services.conversation_timing import history_report
                    await durable_call(history_report, ws)
                    indexed[str(ws.root)] = fingerprint
                await tick(runtime.workspace)
            except Exception:
                logging.exception('Фоновое обучение: ошибка очереди/запуска')

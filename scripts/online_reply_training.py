"""Independent, preemptible incremental MLX worker (one cycle <=30 minutes)."""

import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

from mybot.services.full_local_training import train_full
from mybot.services.local_training import MODEL, PROJECT
from mybot.services.night_training_control import NightGuard, BudgetExpired
from mybot.services.online_learning import read_state, state_path, queue_lock
from mybot.services.style_versions import activate, selected
from mybot.storage.atomic import read_json, write_json
from scripts.train_reply_adapter import resolve_workspace


def waiting_gpu(ws, root):
    run = read_json(root / 'run.json', {})
    resumable = (root / 'checkpoint.json').exists()
    run.update(account_id=ws.account_id, dialog_id=ws.dialog_id, pid=os.getpid(),
               status='paused' if resumable else 'waiting_gpu', phase='waiting_gpu')
    run.setdefault('completed_steps', 0)
    write_json(root / 'run.json', run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import re
    if not re.fullmatch('[a-f0-9]{32}', args.run_id):
        parser.error('Некорректный ID запуска.')
    ws = resolve_workspace(args.account, args.dialog)
    root = ws.root / 'learning/mlx_runs' / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (ws.root / 'learning/online_worker.lock').open('a') as scheduler, (PROJECT / 'local_models/.training.lock').open('a') as gpu:
        try:
            fcntl.flock(scheduler, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another scheduler owns its metadata; do not overwrite it.
            return
        write_json(PROJECT / 'local_models/.inference_release.json', {'at': time.time()})
        deadline = time.monotonic() + 2
        while True:
            try:
                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    waiting_gpu(ws, root)
                    return
                time.sleep(.1)
        state = read_state(ws)
        old = read_json(root / 'run.json', {})
        elapsed = old.get('cycle_elapsed_seconds', 0)
        remaining = 1800 - elapsed
        if remaining <= 60:
            old.update(status='interrupted_or_failed', phase='cycle_time_limit')
            write_json(root / 'run.json', old)
            return
        guard = NightGuard(remaining, reserve=min(300, remaining / 4))
        original_expired = guard.expired
        def expired(training=False):
            if read_json(ws.root / 'learning/online_pause.json', {}).get('pause'):
                guard.stopping = True
            return original_expired(training)
        guard.expired = expired
        signal.signal(signal.SIGTERM, lambda *_: setattr(guard, 'stopping', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(guard, 'stopping', True))
        caffeine = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())]) if sys.platform == 'darwin' else None
        try:
            if args.resume and (root / 'checkpoint.json').exists():
                dataset_id = old['dataset_id']
                resume = True
            else:
                from mlx_lm.utils import load_tokenizer
                from mybot.services.online_dataset import prepare
                guard.wait()
                manifest, excluded = prepare(ws, state['queue'], selected(ws), load_tokenizer(MODEL), MODEL)
                if excluded:
                    with queue_lock(ws):
                        state = read_state(ws)
                        invalid = {e['event_id'] for e in excluded}
                        state['queue'] = [e for e in state['queue'] if e['id'] not in invalid]
                        state['excluded'].extend({'id': e['event_id'], 'reason': e['reason']} for e in excluded)
                        write_json(state_path(ws), state)
                if not manifest:
                    old.update(status='interrupted_or_failed', phase='insufficient_eligible_examples')
                    write_json(root / 'run.json', old)
                    with queue_lock(ws):
                        state = read_state(ws)
                        state.pop('run_id', None)
                        state.update(phase='collecting', result='После проверки длины осталось меньше 10 пригодных новых ответов.')
                        write_json(state_path(ws), state)
                    return
                dataset_id = manifest['id']
                resume = False
            train_full(ws, dataset_id, MODEL, run_id=args.run_id, hours=remaining / 3600,
                       resume=resume, guard=guard)
            run = read_json(root / 'run.json')
            with queue_lock(ws):
                # Re-read after training: actual-sent events may have arrived during this cycle.
                state = read_state(ws)
                if run['status'] != 'paused':
                    manifest = read_json(ws.root / 'learning/reviewed' / (dataset_id + '.json'))
                    consumed = set(state['consumed']) | set(manifest['new_events'])
                    state['consumed'] = sorted(consumed)
                    state['queue'] = [e for e in state['queue'] if e['id'] not in consumed]
                    if run.get('quality_status') == 'passed' and not guard.expired(False):
                        activate(ws, args.run_id, automatic=True)
                        state.update(phase='applied', result='Новая версия применена; предыдущая сохранена для отката.')
                    else:
                        state.update(phase='rejected', result='Версия не применена: контрольная ошибка или проверка генерации/указаний не пройдена.')
                else:
                    state.update(phase='paused_for_reply' if guard.stopping else 'paused_time_limit')
                run['cycle_elapsed_seconds'] = elapsed + time.monotonic() - started
                write_json(root / 'run.json', run)
                write_json(state_path(ws), state)
                write_json(root / 'application.json', {
                    'applied': state.get('phase') == 'applied', 'result': state.get('result'),
                    'active_version': selected(ws), 'parent_version': run.get('parent_adapter')})
        except BudgetExpired:
            run = read_json(root / 'run.json', {})
            run.update(status='paused', phase='paused_for_reply' if guard.stopping else 'paused_time_limit',
                       cycle_elapsed_seconds=elapsed + time.monotonic() - started)
            run.pop('error', None)
            write_json(root / 'run.json', run)
            with queue_lock(ws):
                state = read_state(ws)
                state.update(phase=run['phase'], result=('Обучение приостановлено ради ответа; продолжится после простоя.'
                    if guard.stopping else 'Достигнут предел времени обучения. Текущая версия сохранена.'))
                write_json(state_path(ws), state)
        except Exception as error:
            run = read_json(root / 'run.json', {})
            run.update(status='interrupted_or_failed', phase='failed', error=type(error).__name__,
                       cycle_elapsed_seconds=elapsed + time.monotonic() - started)
            write_json(root / 'run.json', run)
            with queue_lock(ws):
                state = read_state(ws)
                state.update(phase='failed', result='Обучение не применено: ' + type(error).__name__)
                write_json(state_path(ws), state)
            raise
        finally:
            if caffeine:
                caffeine.terminate()
                caffeine.wait()


if __name__ == '__main__':
    main()

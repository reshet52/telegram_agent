"""Validate an existing same-dialog adapter without changing any weights."""

import argparse
import fcntl
import os
import re
import signal

from mybot.services.local_training import MODEL, PROJECT
from mybot.services.generation_check import check
from mybot.storage.atomic import write_json, read_json
from scripts.train_reply_adapter import resolve_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True, type=int)
    parser.add_argument('--dialog', required=True, type=int)
    parser.add_argument('--version', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-path', default=str(MODEL))
    args = parser.parse_args()
    if not all(re.fullmatch(r'[a-f0-9]{32}', x) for x in (args.version, args.run_id)):
        parser.error('Invalid version/job ID')
    ws = resolve_workspace(args.account, args.dialog)
    path = ws.root / 'learning/mlx_runs' / args.run_id / 'run.json'
    run = dict(account_id=ws.account_id, dialog_id=ws.dialog_id, pid=os.getpid(),
               phase='checking_generation', status='checking', candidate=args.version)
    def progress(done, total):
        run.update(checked_answers=done, total_answers=total)
        write_json(path, run)
    def interrupted(*_):
        raise KeyboardInterrupt('Проверка остановлена.')
    signal.signal(signal.SIGTERM, interrupted)
    write_json(path, run)
    try:
        with (PROJECT / 'local_models/.training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            candidate_root = ws.root / 'learning/mlx_runs' / args.version
            candidate = read_json(candidate_root / 'run.json', {})
            if candidate.get('mode') == 'full':
                from mybot.services.full_local_training import compare, write_report
                from mybot.services.night_training_control import NightGuard
                from mybot.services.reviewed_dataset import load_reviewed
                from mybot.services.style_versions import adapter, quality_path
                adapter(ws, args.version, require_quality=False)
                manifest, _ = load_reviewed(ws, candidate['dataset_id'])
                comparison = compare(ws, candidate_root, manifest, candidate, NightGuard(3600, reserve=0))
                candidate.update(quality_status=comparison['status'], phase='done')
                write_json(candidate_root / 'run.json', candidate)
                write_report(candidate_root, manifest, candidate, comparison)
                report = read_json(quality_path(ws, args.version), {})
                if comparison['status'] == 'incomplete_time_limit':
                    raise ValueError('Сравнение сохранено не полностью; повторите проверку для продолжения.')
            else:
                report = check(ws, args.version, args.model_path, progress=progress)
        run.update(status='checked', quality_status=report['status'], counts=report['counts'],
                   expected_per_model=report['expected_per_model'])
    except BaseException as error:
        run.update(status='interrupted_or_failed', error=str(error)[:300] if isinstance(error, ValueError) else type(error).__name__)
        write_json(path, run)
        raise
    write_json(path, run)


if __name__ == '__main__':
    main()

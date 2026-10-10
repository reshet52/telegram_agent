"""Launch/resume an authorized full-history run independently of this terminal."""

import argparse
import os
import re
import subprocess
from uuid import uuid4

from mybot.services.local_training import MODEL, PROJECT, PYTHON, available
from mybot.services.reviewed_dataset import load_reviewed
from mybot.services.training_jobs import status, alive
from mybot.services.style_versions import selected
from mybot.storage.atomic import read_json, write_json
from mybot.storage.style_lesson import StyleLesson
from scripts.train_reply_adapter import resolve_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True, type=int)
    parser.add_argument('--dialog', required=True, type=int)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--resume', help='Existing run ID with a complete checkpoint.')
    args = parser.parse_args()
    if not 0 < args.hours <= 24 or args.resume and not re.fullmatch(r'[a-f0-9]{32}', args.resume):
        parser.error('Некорректный предел времени/ID.')
    if not available():
        raise ValueError('Локальная модель или MLX недоступны.')
    ws = resolve_workspace(args.account, args.dialog)
    previous = status(ws)
    if previous and alive(previous):
        raise ValueError('Уже идёт задача этого диалога.')
    manifest, _ = load_reviewed(ws, args.dataset)
    if manifest.get('version') != 3:
        raise ValueError('Нужен полный набор версии 3.')
    key = args.resume or uuid4().hex
    root = ws.root / 'learning/mlx_runs' / key
    if args.resume:
        run = read_json(root / 'run.json', {})
        if run.get('dataset_id') != args.dataset or not (root / 'checkpoint.json').is_file():
            raise ValueError('Нет контрольной точки этого набора.')
    else:
        root.mkdir(parents=True, exist_ok=False)
    job = {'id': key, 'dataset_id': args.dataset, 'full': True, 'hours': args.hours,
           'baseline': selected(ws), 'state': 'preparing', 'pid': None}
    lesson = StyleLesson(ws)
    lesson.write('training_job.json', job)
    command = [str(PYTHON), '-m', 'scripts.train_reply_mlx', '--full', '--account', str(args.account),
               '--dialog', str(args.dialog), '--dataset', args.dataset, '--model-path', str(MODEL),
               '--run-id', key, '--hours', str(args.hours)]
    if args.resume:
        command.append('--resume')
    with (root / 'process.log').open('a' if args.resume else 'w') as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1'))
    job['pid'] = process.pid
    lesson.write('training_job.json', job)
    print('Run:', key)
    print('PID:', process.pid)
    print('Report:', root / 'report.md')
    print('Log:', root / 'process.log')


if __name__ == '__main__':
    main()

"""Restart-aware, cancellable local jobs. Telegram callbacks never await training."""

import asyncio
import os
import signal
import subprocess
from uuid import uuid4

from mybot.services.local_training import PROJECT, PYTHON, MODEL, available
from mybot.services.reviewed_dataset import prepare_reviewed, load_reviewed
from mybot.services.style_versions import selected, adapter, practice_selected, quality_passed
from mybot.ai.local_format import FORMAT_VERSION
from mybot.storage.atomic import read_json, write_json
from mybot.storage.style_lesson import StyleLesson

_processes = {}
_waiters = set()


def alive(job):
    process = _processes.get(job['id'])
    if process is not None:
        return process.returncode is None
    pid = job.get('pid')
    if not pid:
        return False
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True,
                                text=True, timeout=2, check=False)
        if getattr(result, 'returncode', 0) and getattr(result, 'stderr', '').strip():
            raise ValueError('Не удалось проверить процесс. Новый запуск заблокирован до повторной проверки.')
        command = result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError('Не удалось проверить работающий процесс. Повторите проверку; новый запуск заблокирован.') from error
    module = {'preview': 'scripts.prepare_style_history', 'check': 'scripts.check_reply_quality'}.get(job.get('kind'), 'scripts.train_reply_mlx')
    return module in command and ('--run-id ' + job['id']) in command


def status(workspace):
    lesson = StyleLesson(workspace)
    job = lesson.read('training_job.json')
    if not job:
        return None
    run = read_json(lesson.root / 'mlx_runs' / job['id'] / 'run.json', {})
    if not job.get('pid') and run.get('pid'):
        job['pid'] = run['pid']
    if alive(job):
        job['state'] = 'stopping' if job.get('state') == 'stopping' else run.get('phase', 'preparing')
    elif run.get('status') == 'trained_unapproved':
        job['state'] = 'ready' if quality_passed(workspace, job['id']) else 'needs_check'
    elif run.get('status') == 'checked':
        job['state'] = 'checked'
    elif run.get('status') == 'prepared':
        job['state'] = 'preview_ready'
        job['dataset_id'] = run['dataset_id']
    elif run.get('status') == 'paused':
        job['state'] = 'paused'
    elif job.get('state') in {'stopping', 'stopped'}:
        job['state'] = 'stopped'
    else:
        job['state'] = 'failed'
    job['run'] = run
    lesson.write('training_job.json', job)
    return job


async def preview_history(workspace, purpose='history', repair_version=None, *, full=False):
    previous = status(workspace)
    if previous and alive(previous):
        return previous
    if (previous and previous.get('state') == 'preview_ready'
            and previous.get('purpose') == purpose and previous.get('repair_version') == repair_version
            and bool(previous.get('full')) == full):
        try:
            manifest, _ = load_reviewed(workspace, previous['dataset_id'])
            if manifest.get('format_version') != FORMAT_VERSION:
                raise ValueError('Старый формат')
        except ValueError:
            pass  # Keep the old manifest, prepare a new version from current sources.
        else:
            return previous
    if not available():
        raise ValueError('Локальная модель или окружение MLX недоступны.')
    # Reserve once under the caller's UI lock, before the worker and lessons can run concurrently.
    if not full:
        StyleLesson(workspace).holdout(create=True)
    key = uuid4().hex
    job = {'id': key, 'kind': 'preview', 'purpose': purpose, 'repair_version': repair_version,
           'full': full, 'state': 'preparing', 'pid': None}
    arguments = ['--purpose', purpose] + (['--repair-version', repair_version] if repair_version else [])
    if full:
        arguments.append('--full')
    return await spawn(workspace, job, 'scripts.prepare_style_history', arguments)


async def check_version(workspace, version):
    previous = status(workspace)
    if previous and alive(previous):
        return previous
    if not available():
        raise ValueError('Локальная модель или окружение MLX недоступны.')
    adapter(workspace, version, require_quality=False)
    job = {'id': uuid4().hex, 'kind': 'check', 'candidate': version, 'state': 'preparing', 'pid': None}
    return await spawn(workspace, job, 'scripts.check_reply_quality', ['--version', version])


async def start(workspace, dataset_id=None, *, hours=8):
    previous = status(workspace)
    if previous and alive(previous):
        return previous
    if not available():
        raise ValueError('Локальная модель или окружение MLX недоступны.')
    lesson = StyleLesson(workspace)
    lesson.holdout(create=True)
    if dataset_id:
        if not previous or previous.get('state') != 'preview_ready' or previous.get('dataset_id') != dataset_id:
            raise ValueError('Подготовка устарела. Подготовьте набор заново.')
        manifest, _ = load_reviewed(workspace, dataset_id)
        if manifest.get('format_version') != FORMAT_VERSION:
            raise ValueError('Нужно заново подготовить набор в исправленном формате.')
        baseline = 'base' if manifest.get('purpose') == 'history' else practice_selected(workspace)
    else:
        manifest = prepare_reviewed(workspace)
        baseline = selected(workspace)
    if baseline not in {'base', 'configured'}:
        adapter(workspace, baseline)
    key = uuid4().hex
    job = {'id': key, 'dataset_id': manifest['id'], 'baseline': 'base' if baseline == 'configured' else baseline,
           'state': 'preparing', 'pid': None}
    arguments = ['--dataset', manifest['id']]
    if manifest.get('version') == 3:
        job.update(full=True, hours=hours)
        arguments += ['--full', '--hours', str(hours)]
    return await spawn(workspace, job, 'scripts.train_reply_mlx', arguments)


async def resume_full(workspace, run_id, *, hours=8):
    import re
    if not re.fullmatch(r'[a-f0-9]{32}', run_id) or not 0 < hours <= 24:
        raise ValueError('Некорректный запуск или предел времени.')
    previous = status(workspace)
    if previous and alive(previous):
        raise ValueError('Уже идёт локальная задача.')
    root = workspace.root / 'learning/mlx_runs' / run_id
    run = read_json(root / 'run.json', {})
    if (run.get('mode') != 'full' or (run.get('account_id'), run.get('dialog_id')) !=
            (workspace.account_id, workspace.dialog_id) or not (root / 'checkpoint.json').is_file()):
        raise ValueError('Нет полной контрольной точки этого диалога.')
    load_reviewed(workspace, run['dataset_id'])
    job = {'id': run_id, 'dataset_id': run['dataset_id'], 'full': True, 'hours': hours,
           'baseline': run['baseline'], 'state': 'preparing', 'pid': None}
    return await spawn(workspace, job, 'scripts.train_reply_mlx',
                       ['--full', '--resume', '--dataset', run['dataset_id'], '--hours', str(hours)])


async def spawn(workspace, job, module, arguments):
    from mybot.ai.local_worker import worker
    await worker.release()
    lesson = StyleLesson(workspace)
    key = job['id']
    root = lesson.root / 'mlx_runs' / key
    resuming = '--resume' in arguments
    root.mkdir(parents=True, exist_ok=resuming)
    lesson.write('training_job.json', job)
    with (root / 'process.log').open('a' if resuming else 'w') as log:
        process = await asyncio.create_subprocess_exec(str(PYTHON), '-m', module,
            '--account', str(workspace.account_id), '--dialog', str(workspace.dialog_id),
            '--model-path', str(MODEL), '--run-id', key, *arguments,
            cwd=PROJECT, env=dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1'),
            stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    _processes[key] = process
    job['pid'] = process.pid
    lesson.write('training_job.json', job)
    task = asyncio.create_task(process.wait())
    _waiters.add(task)
    task.add_done_callback(_waiters.discard)
    return job


def stop(workspace):
    lesson = StyleLesson(workspace)
    job = status(workspace)
    if job and alive(job):
        try:
            os.kill(job['pid'], signal.SIGTERM)
        except ProcessLookupError:
            return status(workspace)
        job['state'] = 'stopping'
        lesson.write('training_job.json', job)
        task = asyncio.create_task(force_stop(job))
        _waiters.add(task)
        task.add_done_callback(_waiters.discard)
    return job


async def force_stop(job):
    await asyncio.sleep(10)
    if alive(job):
        try:
            os.kill(job['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass

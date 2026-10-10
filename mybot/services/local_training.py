"""Bounded subprocess training, without loading MLX into the bot."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

from mybot.storage.atomic import read_json

PROJECT = Path(__file__).resolve().parents[2]
PYTHON = PROJECT / '.venv-mlx' / 'bin' / 'python'
MODEL = PROJECT / 'local_models' / 'qwen3-4b-instruct-4bit'
_running = False


def available():
    return PYTHON.is_file() and (MODEL / 'config.json').is_file() and any(MODEL.glob('*.safetensors'))


async def run_local_training(workspace, dataset_id):
    global _running
    if _running:
        raise ValueError('Локальное обучение уже идёт. Дождитесь завершения.')
    if not available():
        raise ValueError('Не найдено окружение MLX или полностью скачанная модель.')
    _running = True
    process = None
    log_path = workspace.root / 'learning' / 'mlx_logs' / f'{uuid4().hex}.log'
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                   PYTHONUNBUFFERED='1')
        with log_path.open('w') as log:
            from mybot.ai.local_worker import worker
            await worker.release()
            process = await asyncio.create_subprocess_exec(
                str(PYTHON), '-m', 'scripts.train_reply_mlx', '--account', str(workspace.account_id),
                '--dialog', str(workspace.dialog_id), '--dataset', dataset_id,
                '--model-path', str(MODEL), '--steps', '50',
                cwd=PROJECT, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
            await asyncio.wait_for(process.wait(), timeout=3600)
        if process.returncode:
            raise ValueError('Обучение не завершилось. Подробности сохранены локально: '
                             f'{log_path}. Рабочая модель не переключена.')
        # The runner's last line is its unique output directory, not a latest-run lookup.
        root = Path(log_path.read_text().splitlines()[-1])
        root = root if root.is_absolute() else PROJECT / root
        expected = (workspace.root / 'learning' / 'mlx_runs').resolve()
        if root.resolve().parent != expected:
            raise ValueError('Некорректный результат процесса обучения.')
        run = read_json(root / 'run.json')
        if (not run or run.get('status') != 'trained_unapproved'
                or run.get('account_id') != workspace.account_id
                or run.get('dialog_id') != workspace.dialog_id
                or run.get('dataset_id') != dataset_id):
            raise ValueError('Результат обучения не прошёл проверку.')
        return run
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        _running = False

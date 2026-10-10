"""Explicit local QLoRA training on owner-reviewed examples (Apple Silicon)."""

import argparse
import fcntl
import hashlib
import math
import os
import signal
import time
from pathlib import Path
from uuid import uuid4

from mybot.services.reviewed_dataset import load_reviewed, MIN_TRAIN, MIN_VALID
from mybot.storage.atomic import write_json
from mybot.storage.reply_practice import now
from scripts.train_reply_adapter import resolve_workspace


def train(workspace, dataset_id, model_path, *, steps=50, max_length=1024, run_id=None):
    started = time.monotonic()
    if type(steps) is not int or not 1 <= steps <= 200 or max_length not in (512, 1024, 2048):
        raise ValueError('Некорректные пределы пробного обучения.')
    model_path = Path(model_path).resolve()
    if not (model_path / 'config.json').is_file() or not list(model_path.glob('*.safetensors')):
        raise ValueError('Локальная модель не скачана полностью.')
    manifest, rows = load_reviewed(workspace, dataset_id)
    from mybot.ai.local_format import FORMAT_VERSION, FinalAnswerDataset
    if manifest.get('format_version') != FORMAT_VERSION:
        raise ValueError('Старый формат набора. Подготовьте набор заново перед обучением.')
    prepared = manifest.get('version') == 2
    if prepared:
        from mybot.services.history_training import tokenizer_hash
        if manifest['tokenizer_hash'] != tokenizer_hash(model_path):
            raise ValueError('Токенизатор изменился. Подготовьте набор заново.')
        max_length = manifest['max_length']
        steps = len(rows['train'])  # Batch size 1: exactly one complete epoch.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    # Dependencies stay out of the Telegram process. No cloud fallback.
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train as mlx_train
    from mlx_lm.tuner.callbacks import TrainingCallback
    from mlx_lm.tuner.utils import linear_to_lora_layers

    np.random.seed(42)
    mx.random.seed(42)
    model, tokenizer = load(str(model_path))
    datasets = {}
    for split, items in rows.items():
        chat = FinalAnswerDataset(items, tokenizer)
        processed = chat.items
        # Never truncate the answer or train on prompt-only examples.
        kept = [chat[i] for i, (tokens, offset) in enumerate(processed)
                if 0 < offset < len(tokens) <= max_length]
        if prepared and len(kept) != len(items):
            raise ValueError('Проверка длины расходится с подготовленным набором. Обучение не начато.')
        if len(kept) < (MIN_TRAIN if split == 'train' else MIN_VALID):
            raise ValueError(f'После ограничения длины недостаточно примеров: {split}={len(kept)}.')
        datasets[split] = kept

    root = workspace.root / 'learning' / 'mlx_runs' / (run_id or uuid4().hex)
    root.mkdir(parents=True, exist_ok=bool(run_id))
    config = {'fine_tune_type': 'lora', 'num_layers': 4,
              'lora_parameters': {'rank': 8, 'scale': 16.0, 'dropout': 0.0,
                                  'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
    write_json(root / 'adapter_config.json', config)
    run = {'version': 1, 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
           'pid': os.getpid(),
           'dataset_id': dataset_id, 'dataset_hash': manifest['hash'],
           'base_model': str(model_path), 'created_at': now(), 'status': 'training',
           'steps': steps, 'max_length': max_length,
           'purpose': manifest.get('purpose', 'reviewed'), 'parent_adapter': manifest.get('parent_adapter'),
           'epochs': 1 if prepared else None,
           'format_version': FORMAT_VERSION, 'learning_rate': 2e-5,
           'counts': {name: len(items) for name, items in datasets.items()}}
    write_json(root / 'run.json', run)
    try:
        model.freeze()
        linear_to_lora_layers(model, config['num_layers'], config['lora_parameters'])
        if manifest.get('parent_adapter'):
            from mybot.services.style_versions import adapter
            parent = adapter(workspace, manifest['parent_adapter'])
            from mybot.storage.atomic import read_json
            parent_run = read_json(parent / 'run.json')
            if parent_run.get('purpose') != 'history' or parent_run['base_model'] != str(model_path):
                raise ValueError('Исходный адаптер не совместим с этим обучением.')
            model.load_weights(str(parent / 'adapters.safetensors'), strict=False)
            run['parent_sha256'] = parent_run['adapter_sha256']

        def fingerprint():
            value = hashlib.sha256()
            for name, tensor in tree_flatten(model.trainable_parameters()):
                value.update(name.encode())
                value.update(np.array(tensor.astype(mx.float32)).tobytes())
            return value.hexdigest()

        run['trainable_parameters'] = sum(t.size for _, t in tree_flatten(model.trainable_parameters()))
        if not run['trainable_parameters']:
            raise ValueError('У модели не найдены обучаемые слои.')
        run['weights_before'] = fingerprint()
        if manifest.get('parent_adapter') and run['weights_before'] != parent_run['weights_after']:
            raise ValueError('Не удалось загрузить веса первоначального адаптера.')
        eval_args = dict(batch_size=1, num_batches=len(datasets['valid']), max_seq_length=max_length)
        run['phase'] = 'checking_before'
        write_json(root / 'run.json', run)
        run['loss_before'] = evaluate(model, datasets['valid'], **eval_args)
        run['phase'] = 'training'
        write_json(root / 'run.json', run)
        class Progress(TrainingCallback):
            def on_train_loss_report(self, info):
                run['completed_steps'] = info['iteration']
                run['peak_memory_gb'] = info['peak_memory']
                write_json(root / 'run.json', run)
        mlx_train(model, optim.Adam(learning_rate=2e-5), datasets['train'],
                  args=TrainingArgs(batch_size=1, iters=steps, max_seq_length=max_length,
                      steps_per_report=1, steps_per_save=steps,
                      adapter_file=str(root / 'adapters.safetensors'), grad_checkpoint=True),
                  training_callback=Progress())
        run['phase'] = 'checking_after'
        write_json(root / 'run.json', run)
        run['loss_after'] = evaluate(model, datasets['valid'], **eval_args)
        run['weights_after'] = fingerprint()
        if (not all(math.isfinite(run[k]) for k in ('loss_before', 'loss_after'))
                or run['weights_before'] == run['weights_after']):
            raise ValueError('Проверка обновления весов не пройдена.')
        # Revalidate revoked feedback or removed source messages after the long run.
        load_reviewed(workspace, dataset_id)
        run['adapter_sha256'] = hashlib.sha256((root / 'adapters.safetensors').read_bytes()).hexdigest()
        run.update(status='trained_unapproved', finished_at=now(),
                   completed_steps=steps, elapsed_seconds=time.monotonic() - started,
                   validation_improved=run['loss_after'] < run['loss_before'])
    except BaseException:
        run.update(status='interrupted_or_failed', finished_at=now())
        write_json(root / 'run.json', run)
        raise
    write_json(root / 'run.json', run)
    del model, tokenizer, datasets
    mx.clear_cache()
    from mybot.services.generation_check import check
    run['phase'] = 'checking_generation'
    write_json(root / 'run.json', run)
    def generation_progress(done, total):
        run.update(checked_answers=done, total_answers=total)
        write_json(root / 'run.json', run)
    report = check(workspace, root.name, model_path, progress=generation_progress)
    run.update(quality_status=report['status'], phase='done')
    write_json(root / 'run.json', run)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--run-id')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--max-steps', type=int, help='Diagnostic cap for full-mode verification.')
    parser.add_argument('--skip-comparison', action='store_true', help='Diagnostic runs only.')
    args = parser.parse_args()
    workspace = resolve_workspace(args.account, args.dialog)
    if not 0 < args.hours <= 24 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('Некорректный предел времени/шагов.')
    if args.resume and (not args.full or not args.run_id):
        parser.error('--resume требует --full и --run-id')
    if not args.full and (args.max_steps or args.skip_comparison):
        parser.error('Диагностические параметры требуют --full')
    if args.run_id:
        import re
        if not re.fullmatch(r'[a-f0-9]{32}', args.run_id):
            parser.error('Некорректный ID запуска.')
    guard = None
    if args.full:
        from mybot.services.night_training_control import NightGuard
        guard = NightGuard(args.hours * 3600)
    def interrupted(*_):
        if guard:
            guard.stopping = True
        else:
            raise KeyboardInterrupt('Обучение остановлено.')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.run_id and not args.full:
        write_json(workspace.root / 'learning' / 'mlx_runs' / args.run_id / 'run.json',
                   {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
                    'pid': os.getpid(), 'status': 'training', 'phase': 'preparing'})
    # OS lock is released even after a crash; prevents two bot/CLI training jobs.
    lock_path = Path(__file__).resolve().parents[1] / 'local_models' / '.training.lock'
    lock_path.parent.mkdir(exist_ok=True)
    write_json(lock_path.parent / '.inference_release.json', {'at': time.time()})
    with lock_path.open('a') as lock:
        try:
            deadline = time.monotonic() + 2
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
        except BlockingIOError:
            if args.run_id:
                write_json(workspace.root / 'learning' / 'mlx_runs' / args.run_id / 'run.json',
                           {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
                            'pid': os.getpid(), 'status': 'interrupted_or_failed',
                            'error': 'Локальная модель занята другой задачей. Повторите после её завершения.'})
            raise SystemExit('Локальная модель занята другой задачей. Повторите после её завершения.')
        try:
            if args.full:
                from mybot.services.full_local_training import train_full
                import subprocess
                import sys
                # Sleep inhibition lasts exactly as long as this independent worker.
                caffeine = (subprocess.Popen(['/usr/bin/caffeinate', '-i', '-s', '-w', str(os.getpid())])
                            if sys.platform == 'darwin' else None)
                try:
                    result = train_full(workspace, args.dataset, args.model_path,
                        run_id=args.run_id or uuid4().hex, hours=args.hours, resume=args.resume,
                        guard=guard, max_steps=args.max_steps, skip_comparison=args.skip_comparison)
                    print(result, flush=True)
                finally:
                    if caffeine:
                        caffeine.terminate()
                        caffeine.wait()
            else:
                print(train(workspace, args.dataset, args.model_path, steps=args.steps, run_id=args.run_id), flush=True)
        except BaseException as error:
            if args.run_id:
                from mybot.storage.atomic import read_json
                path = workspace.root / 'learning' / 'mlx_runs' / args.run_id / 'run.json'
                run = read_json(path, {})
                run.update(account_id=workspace.account_id, dialog_id=workspace.dialog_id,
                           status='trained_unapproved' if run.get('adapter_sha256') else 'interrupted_or_failed',
                           quality_status='interrupted_or_failed',
                           error=str(error)[:400] if isinstance(error, ValueError) else type(error).__name__)
                write_json(path, run)
            raise


if __name__ == '__main__':
    main()

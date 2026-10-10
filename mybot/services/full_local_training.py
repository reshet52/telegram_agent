"""Resumable full-history MLX training. Imported only in the worker process."""

import hashlib
import math
import os
import random
import time
from pathlib import Path
from uuid import uuid4

from mybot.ai.local_format import FORMAT_VERSION, training_tokens
from mybot.services.history_training import tokenizer_hash
from mybot.services.night_training_control import BudgetExpired, NightGuard
from mybot.services.reviewed_dataset import load_reviewed
from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.reply_practice import now


LORA_CONFIG = {'fine_tune_type': 'lora', 'num_layers': 4,
               'lora_parameters': {'rank': 8, 'scale': 16.0, 'dropout': 0.0,
                                   'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def model_hash(path):
    return {p.name: file_hash(p) for p in sorted(Path(path).glob('*.safetensors'))}


def save_checkpoint(root, model, optimizer, run, *, mx, tree_flatten):
    """Publish a complete generation atomically; old generations remain recoverable."""
    import numpy as np
    folder = root / 'checkpoints' / (str(run['completed_steps']) + '_' + uuid4().hex)
    folder.mkdir(parents=True)
    mx.eval(model.trainable_parameters(), optimizer.state, mx.random.state)
    mx.save_safetensors(str(folder / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
    mx.savez(str(folder / 'optimizer.npz'), **dict(tree_flatten(optimizer.state)))
    mx.savez(str(folder / 'random.npz'), **{str(i): x for i, x in enumerate(mx.random.state)})
    numpy_state = np.random.get_state()
    state = {'run': run, 'python_random': random.getstate(),
             'numpy_random': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
             'files': {name: file_hash(folder / name) for name in
                       ('adapters.safetensors', 'optimizer.npz', 'random.npz')}}
    write_json(folder / 'state.json', state)
    # Durably flush binary files before publishing the pointer.
    for name in state['files']:
        with (folder / name).open('rb') as stream:
            os.fsync(stream.fileno())
    write_json(root / 'checkpoint.json', {'folder': folder.name,
                                         'state_sha256': file_hash(folder / 'state.json')})
    return folder


def restore_checkpoint(root, model, optimizer, *, mx, tree_unflatten):
    import numpy as np
    pointer = read_json(root / 'checkpoint.json')
    if not pointer or Path(pointer['folder']).name != pointer['folder']:
        raise ValueError('Нет корректной контрольной точки; старт с нуля запрещён.')
    folder = root / 'checkpoints' / pointer['folder']
    if file_hash(folder / 'state.json') != pointer['state_sha256']:
        raise ValueError('Повреждена контрольная точка.')
    state = read_json(folder / 'state.json')
    if any(file_hash(folder / name) != fingerprint for name, fingerprint in state['files'].items()):
        raise ValueError('Повреждены веса или состояние оптимизатора.')
    model.load_weights(str(folder / 'adapters.safetensors'), strict=False)
    optimizer.state = tree_unflatten(list(mx.load(str(folder / 'optimizer.npz')).items()))
    optimizer.init(model.trainable_parameters())
    arrays = mx.load(str(folder / 'random.npz'))
    # MLX 0.32 uses a read-only random-state sentinel. Its Threefry key is
    # exactly the high/low uint32 halves of seed(uint64), not a hash of the seed.
    if len(arrays) != 1 or arrays['0'].shape != (2,) or arrays['0'].dtype != mx.uint32:
        raise ValueError('Неподдерживаемое состояние генератора MLX.')
    key = arrays['0'].tolist()
    mx.random.seed((int(key[0]) << 32) | int(key[1]))
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    random.setstate(tuples(state['python_random']))
    n = state['numpy_random']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    return state['run']


def write_report(root, manifest, run, comparison=None):
    online = manifest.get('version') == 4
    report = {'account_id': run['account_id'], 'dialog_id': run['dialog_id'],
              'status': run['status'], 'phase': run.get('phase'),
              'coverage_counts': manifest['coverage_counts'], 'counts': manifest['counts'],
              'completed_steps': run['completed_steps'], 'planned_steps': run['steps'],
              'best_step': run.get('best_step'), 'best_loss': run.get('best_loss'),
              'loss_before': run.get('loss_before'), 'evaluations': run.get('evaluations', []),
              'pause_seconds': run.get('pause_seconds', 0), 'pause_count': run.get('pause_count', 0),
              'stop_reason': run.get('stop_reason'), 'error': run.get('error'),
              'comparison': comparison, 'live_model_changed': False,
              'resume_command': (f'.venv-mlx/bin/python -m scripts.train_reply_mlx --full '
                  f'--account {run["account_id"]} --dialog {run["dialog_id"]} '
                  f'--model-path {run["base_model"]} --dataset {run["dataset_id"]} '
                  f'--run-id {root.name} --resume --hours 8')}
    if online:
        report['resume_command'] = (f'.venv-mlx/bin/python -m scripts.online_reply_training '
            f'--account {run["account_id"]} --dialog {run["dialog_id"]} --run-id {root.name} --resume')
    write_json(root / 'report.json', report)
    lines = ['# Постоянное обучение' if online else '# Ночное обучение', '', f'Состояние: {run["status"]}',
             f'Шаги: {run["completed_steps"]}/{run["steps"]}',
             f'Примеры: обучение {manifest["counts"]["train"]}, проверка {manifest["counts"]["valid"]}',
             f'Лучшая версия: шаг {run.get("best_step")}, ошибка {run.get("best_loss")}',
             f'Паузы: {run.get("pause_count", 0)}, {run.get("pause_seconds", 0):.0f} сек.',
             f'Причина остановки: {run.get("stop_reason", "—")}', '',
             'Действующая модель не переключена.', '', '## Покрытие истории', '']
    lines += [f'- {key}: {value}' for key, value in manifest['coverage_counts'].items()]
    if comparison:
        lines += ['', '## Сравнение', '', f'Состояние: {comparison["status"]}',
                  'Тексты в comparison.json; техническая проверка не оценивает сходство стиля.']
        for index, case in enumerate(comparison['results'], 1):
            lines += ['', f'### Ситуация {index}', '']
            lines += [f'{m["sender"]}: {m["text"]}' for m in case['context']]
            for side in ('current', 'candidate'):
                if case.get(side):
                    lines += ['', f'{side}:', '', case[side]['text'],
                              f'Проверка: {case[side]["rejection"] or "ok"}']
    lines += ['', '## Продолжение', '', '```sh', report['resume_command'], '```']
    write_text(root / 'report.md', '\n'.join(lines))


def compare(workspace, root, manifest, run, guard):
    """Compare the selected old adapter with the best new adapter, never activate."""
    import mlx.core as mx
    from mlx_lm import load
    from mybot.ai.local_reply import live_messages
    from mybot.ai.local_sampling import sample
    from mybot.services.style_versions import adapter, quality_path
    from mybot.services.learning_dataset import example
    from mybot.storage.style_lesson import StyleLesson
    lesson = StyleLesson(workspace)
    cases = lesson.holdout_for(manifest['holdout_ref'])['cases']
    report = read_json(root / 'comparison.json') or {
        'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
        'candidate_sha256': run['adapter_sha256'], 'baseline': run['baseline'],
        'seed': 42, 'status': 'checking', 'results': [], 'quality_assessed': False}
    if report['candidate_sha256'] != run['adapter_sha256']:
        report['results'] = []
        report['candidate_sha256'] = run['adapter_sha256']
    old_path = adapter(workspace, run['baseline']) if run['baseline'] != 'configured' else None
    report['baseline_sha256'] = file_hash(old_path / 'adapters.safetensors') if old_path else None
    try:
        for side, path in (('current', old_path), ('candidate', root)):
            if all(any(r['case_id'] == key and r.get(side) for r in report['results']) for key in cases):
                continue
            guard.wait(training=False)
            model, tokenizer = load(run['base_model'], adapter_path=str(path) if path else None)
            try:
                for key in cases:
                    result = next((r for r in report['results'] if r['case_id'] == key), None)
                    if result and result.get(side):
                        continue
                    guard.wait(training=False)
                    context, expected = lesson.store.source(lesson.store.get(key))
                    messages = live_messages('', context)
                    if messages != example(context, [])['prompt']:
                        raise ValueError('Учебный формат расходится с живым.')
                    _, receipts = sample(model, tokenizer, messages, report['seed'], count=1,
                                         attempts=1, should_stop=lambda: guard.expired(False))
                    if not result:
                        result = {'case_id': key, 'context': context,
                                  'actual_reply': '\n\n'.join(m['text'] for m in expected)}
                        report['results'].append(result)
                    result[side] = receipts[0]
                    write_json(root / 'comparison.json', report)
                    run.update(checked_answers=sum(bool(r.get(s)) for r in report['results']
                                                   for s in ('current', 'candidate')), total_answers=40)
                    write_json(root / 'run.json', run)
            finally:
                del model, tokenizer
                mx.clear_cache()
        load_reviewed(workspace, manifest['id'])
        if file_hash(root / 'adapters.safetensors') != run['adapter_sha256']:
            raise ValueError('Адаптер изменился во время сравнения.')
        good = sum(r['candidate']['rejection'] is None for r in report['results'])
        report.update(status='passed' if good == 20 else 'failed', finished_at=now())
        # Compatible technical gate for the existing explicit activation UI.
        quality = {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
                   'version': root.name, 'format_version': FORMAT_VERSION,
                   'tokenizer_hash': manifest['tokenizer_hash'], 'adapter_sha256': run['adapter_sha256'],
                   'cases': cases, 'status': report['status'],
                   'counts': {'candidate': good,
                              'baseline': sum(r['current']['rejection'] is None for r in report['results'])},
                   'expected_per_model': 20, 'quality_assessed': False,
                   'method': '20_independent_cases_one_seed_current_vs_candidate',
                   'results': report['results']}
        write_json(quality_path(workspace, root.name), quality)
    except (BudgetExpired, TimeoutError):
        report['status'] = 'incomplete_time_limit'
    write_json(root / 'comparison.json', report)
    return report


def check_online(workspace, root, manifest, run, guard):
    """Fail closed on validation regression, broken output or ignored explicit instructions."""
    import mlx.core as mx
    from mlx_lm import load
    from mybot.ai.local_reply import live_messages
    from mybot.ai.local_sampling import sample
    from mybot.services.style_versions import quality_path
    from mybot.storage.style_lesson import StyleLesson
    lesson = StyleLesson(workspace)
    cases = lesson.holdout_for(manifest['holdout_ref'])['cases'][:2]
    report = {'method': 'online_validation_and_instructions_v1', 'status': 'failed',
              'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
              'adapter_sha256': run['adapter_sha256'], 'tokenizer_hash': manifest['tokenizer_hash'],
              'format_version': FORMAT_VERSION, 'cases': cases, 'results': []}
    model, tokenizer = load(run['base_model'], adapter_path=str(root))
    try:
        finite = True
        for tensor in mx.load(str(root / 'adapters.safetensors')).values():
            guard.wait(training=False)
            finite = finite and bool(mx.all(mx.isfinite(tensor)).item())
        for key in cases:
            context, _ = lesson.store.source(lesson.store.get(key))
            for instruction in ('', 'Ответь одним сообщением. Обязательно включи точную фразу «проверю завтра».'):
                guard.wait(training=False)
                facts = __import__('json').dumps({'instruction': instruction}, ensure_ascii=False) if instruction else ''
                variants, receipts = sample(model, tokenizer, live_messages(facts, context), 42, count=1,
                                            should_stop=lambda: guard.expired(False))
                passed = bool(variants and receipts[0]['rejection'] is None and
                              (not instruction or 'проверю завтра' in variants[0].lower()))
                report['results'].append({'case_id': key, 'instruction': instruction,
                                          'passed': passed, 'receipt': receipts[0]})
        report['finite_weights'] = finite
        report['validation_passed'] = (math.isfinite(run['loss_after']) and
                                       run['loss_after'] <= run['loss_before'])
        if finite and report['validation_passed'] and len(report['results']) == 4 and all(r['passed'] for r in report['results']):
            report['status'] = 'passed'
    finally:
        del model, tokenizer
        mx.clear_cache()
    load_reviewed(workspace, manifest['id'])
    write_json(quality_path(workspace, root.name), report)
    return report


def train_full(workspace, dataset_id, model_path, *, run_id, hours=8, resume=False,
               guard=None, max_steps=None, skip_comparison=False):
    started = time.monotonic()
    guard = guard or NightGuard(hours * 3600)
    model_path = Path(model_path).resolve()
    manifest, rows = load_reviewed(workspace, dataset_id)
    if manifest.get('version') not in (3, 4) or manifest['tokenizer_hash'] != tokenizer_hash(model_path):
        raise ValueError('Нужен полный набор с неизменённым токенизатором.')
    online = manifest.get('version') == 4
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm import load
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mybot.services.style_versions import selected, adapter

    root = workspace.root / 'learning' / 'mlx_runs' / run_id
    root.mkdir(parents=True, exist_ok=True)
    previous = read_json(root / 'run.json', {})
    if previous.get('completed_steps', 0) and not resume:
        raise ValueError('Запуск уже содержит обучение. Используйте --resume.')
    run = {'version': 2, 'mode': 'full', 'account_id': workspace.account_id,
           'dialog_id': workspace.dialog_id, 'dataset_id': dataset_id, 'dataset_hash': manifest['hash'],
           'base_model': str(model_path), 'base_model_hashes': model_hash(model_path),
           'created_at': now(), 'status': 'training', 'phase': 'loading', 'pid': os.getpid(),
           'steps': len(manifest['order']), 'completed_steps': 0, 'format_version': FORMAT_VERSION,
           'max_length': manifest['max_length'], 'purpose': 'online' if online else 'history', 'parent_adapter': manifest.get('parent_adapter'),
           'learning_rate': 2e-5, 'counts': manifest['counts'], 'baseline': selected(workspace),
           'evaluations': [], 'best_loss': None, 'best_step': None, 'bad_evaluations': 0,
           'epochs': 'full_history_then_recent_third', 'lora_config': LORA_CONFIG,
           'hours': hours, 'pause_seconds': 0, 'pause_count': 0}
    if run['baseline'] not in ('configured', 'base'):
        adapter(workspace, run['baseline'])
    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    mx.set_cache_limit(512 * 1024 * 1024)
    model, tokenizer = load(str(model_path))
    model.freeze()
    linear_to_lora_layers(model, LORA_CONFIG['num_layers'], LORA_CONFIG['lora_parameters'])
    if online:
        parent = adapter(workspace, manifest['parent_adapter'])
        parent_run = read_json(parent / 'run.json')
        if parent_run['adapter_sha256'] != manifest['parent_sha256'] or parent_run['base_model'] != str(model_path):
            raise ValueError('Действующая версия изменилась или несовместима.')
        model.load_weights(str(parent / 'adapters.safetensors'), strict=False)
        run['parent_sha256'] = manifest['parent_sha256']
    optimizer = optim.Adam(learning_rate=2e-5)
    optimizer.init(model.trainable_parameters())
    if resume:
        restored = restore_checkpoint(root, model, optimizer, mx=mx, tree_unflatten=tree_unflatten)
        for key in ('account_id', 'dialog_id', 'dataset_id', 'dataset_hash', 'base_model',
                    'base_model_hashes', 'format_version', 'max_length', 'lora_config', 'parent_adapter'):
            if run[key] != restored[key]:
                raise ValueError('Контрольная точка несовместима: ' + key)
        run = dict(restored, pid=os.getpid(), status='training', phase='loading', hours=hours)
        run.pop('stop_reason', None)
        run.pop('error', None)
    write_json(root / 'adapter_config.json', LORA_CONFIG)
    run['trainable_parameters'] = sum(t.size for _, t in tree_flatten(model.trainable_parameters()))
    if not run['trainable_parameters']:
        raise ValueError('Нет обучаемых параметров.')
    datasets = {split: [training_tokens(tokenizer, row) for row in values] for split, values in rows.items()}
    if any(not 0 < offset < len(tokens) < manifest['max_length']
           for values in datasets.values() for tokens, offset in values):
        raise ValueError('Длина или граница ответа изменились.')
    grad_checkpoint(model.layers[0])
    loss_grad = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    def batch(item):
        tokens, offset = item
        size = min(manifest['max_length'], ((len(tokens) + 31) // 32) * 32)
        return (mx.array([tokens + [0] * (size - len(tokens))], dtype=mx.int32),
                mx.array([[offset, len(tokens) - 1]], dtype=mx.int32))

    @__import__('functools').partial(mx.compile, inputs=state, outputs=state)
    def step(arrays):
        (loss, count), gradient = loss_grad(model, *arrays)
        finite = mx.all(mx.stack([mx.all(mx.isfinite(t)) for _, t in tree_flatten(gradient)]))
        optimizer.update(model, gradient)
        return loss, count, finite

    def progress(phase, thermal):
        run.update(phase=phase, thermal_state=thermal, thermal_monitor=guard.monitor,
                   pause_seconds=guard.pause_seconds, pause_count=guard.pause_count,
                   elapsed_seconds=time.monotonic() - started)
        write_json(root / 'run.json', run)
    guard.progress = progress

    def evaluate():
        model.eval()
        total, count = 0., 0
        try:
            for item in datasets['valid']:
                guard.wait()
                value, tokens = default_loss(model, *batch(item))
                mx.eval(value, tokens)
                value, tokens = value.item(), tokens.item()
                if not math.isfinite(value):
                    raise ValueError('Нечисловая контрольная ошибка.')
                total += value * tokens
                count += tokens
                mx.clear_cache()
        finally:
            model.train()
        return total / count

    def checkpoint():
        run.update(pause_seconds=guard.pause_seconds, pause_count=guard.pause_count,
                   elapsed_seconds=time.monotonic() - started)
        save_checkpoint(root, model, optimizer, run, mx=mx, tree_flatten=tree_flatten)
        write_json(root / 'run.json', run)
        write_report(root, manifest, run)

    comparison = None
    try:
        progress('checking_before', guard.reader())
        if 'loss_before' not in run:
            run['loss_before'] = evaluate()
            run['best_loss'] = run['loss_before']
            checkpoint()
        model.train()
        progress('training', guard.reader())
        while run['completed_steps'] < run['steps']:
            guard.wait()
            index = manifest['order'][run['completed_steps']]
            value, tokens, finite = step(batch(datasets['train'][index]))
            mx.eval(state, value, tokens, finite)
            loss = value.item()
            if not math.isfinite(loss) or not finite.item():
                # Preserve the last finite checkpoint instead of committing bad weights.
                raise FloatingPointError('Нечисловая учебная ошибка; используйте последнюю конечную контрольную точку.')
            run.update(completed_steps=run['completed_steps'] + 1, train_loss=loss,
                       peak_memory_gb=mx.get_peak_memory() / 1e9)
            mx.clear_cache()
            if run['completed_steps'] % 10 == 0:
                progress('training', guard.reader())
                print(f"step={run['completed_steps']}/{run['steps']} loss={loss:.4f} "
                      f"memory={run['peak_memory_gb']:.2f}GB", flush=True)
            if run['completed_steps'] % 100 == 0:
                checkpoint()
            if (run['completed_steps'] % (50 if online else 500) == 0 or
                    run['completed_steps'] in (manifest['base_steps'], run['steps'])):
                progress('checking_validation', guard.reader())
                measured = evaluate()
                run['evaluations'].append({'step': run['completed_steps'], 'loss': measured, 'at': now()})
                if measured <= run['best_loss'] if online else measured < run['best_loss']:
                    run.update(best_loss=measured, best_step=run['completed_steps'], bad_evaluations=0)
                    folder = root / ('best_' + uuid4().hex)
                    folder.mkdir()
                    mx.save_safetensors(str(folder / 'adapters.safetensors'),
                                        dict(tree_flatten(model.trainable_parameters())))
                    run['best_checkpoint'] = {'folder': folder.name, 'step': run['best_step'],
                                              'sha256': file_hash(folder / 'adapters.safetensors')}
                    write_json(root / 'best.json', run['best_checkpoint'])
                else:
                    run['bad_evaluations'] += 1
                checkpoint()
                progress('training', guard.reader())
                if run['bad_evaluations'] >= 3:
                    run['stop_reason'] = 'early_stopping'
                    break
            if max_steps is not None and run['completed_steps'] >= max_steps:
                run['stop_reason'] = 'diagnostic_step_limit'
                break
        if online and run['completed_steps'] == run['steps'] and not any(
                e['step'] == run['steps'] for e in run['evaluations']):
            # A reply may have interrupted final validation after the last training step.
            measured = evaluate()
            run['evaluations'].append({'step': run['steps'], 'loss': measured, 'at': now()})
            if measured <= run['best_loss']:
                folder = root / ('best_' + uuid4().hex)
                folder.mkdir()
                mx.save_safetensors(str(folder / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
                run.update(best_loss=measured, best_step=run['steps'], best_checkpoint={
                    'folder': folder.name, 'step': run['steps'], 'sha256': file_hash(folder / 'adapters.safetensors')})
            checkpoint()
        run.setdefault('stop_reason', 'complete')
    except BudgetExpired:
        run['stop_reason'] = 'interrupted' if guard.stopping else 'time_limit'
    except BaseException as error:
        run.update(status='interrupted_or_failed', phase='failed', error=type(error).__name__)
        if not isinstance(error, FloatingPointError):
            checkpoint()
        write_json(root / 'run.json', run)
        write_report(root, manifest, run)
        raise
    run.update(status='paused' if run['completed_steps'] < run['steps'] and
               run['stop_reason'] != 'early_stopping' else 'trained_unapproved', phase='saved', finished_at=now())
    checkpoint()
    best = run.get('best_checkpoint')
    if best:
        path = root / best['folder'] / 'adapters.safetensors'
        if file_hash(path) != best['sha256']:
            raise ValueError('Повреждён лучший адаптер.')
        # Atomic publication of the evaluated best weights, separate from resumable last weights.
        import shutil
        temporary = root / '.best.safetensors'
        shutil.copyfile(path, temporary)
        os.replace(temporary, root / 'adapters.safetensors')
        run.update(adapter_sha256=best['sha256'], validation_improved=True,
                   loss_after=run['best_loss'], status='trained_unapproved')
    del step, loss_grad, state, model, tokenizer, datasets, optimizer
    mx.clear_cache()
    if online and (run['completed_steps'] < run['steps'] or guard.stopping):
        run['status'] = 'paused'
    if best and not skip_comparison and not guard.expired(False) and (not online or run['status'] != 'paused'):
        progress('checking_generation', guard.reader())
        try:
            comparison = (check_online(workspace, root, manifest, run, guard) if online
                          else compare(workspace, root, manifest, run, guard))
            run['quality_status'] = comparison['status']
        except Exception as error:
            run['quality_status'] = 'interrupted_or_failed'
            run['comparison_error'] = type(error).__name__
    if online and (guard.stopping or (best and not run.get('quality_status') and guard.expired(False))):
        run['status'] = 'paused'
    run.update(phase='done', elapsed_seconds=time.monotonic() - started,
               pause_seconds=guard.pause_seconds, pause_count=guard.pause_count)
    write_json(root / 'run.json', run)
    write_report(root, manifest, run, comparison)
    return root

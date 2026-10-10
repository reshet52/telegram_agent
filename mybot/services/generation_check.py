"""Local deterministic technical gate; not a style-quality score."""

import hashlib
import time

from mybot.ai.local_format import FORMAT_VERSION
from mybot.ai.local_sampling import sample
from mybot.services.learning_dataset import example
from mybot.services.decision_task import messages as decision_messages
from mybot.services.style_versions import adapter, quality_path, activate_practice
from mybot.storage.atomic import write_json, read_json
from mybot.storage.style_lesson import StyleLesson

SEEDS = (42, 314159)


def check(workspace, version, model_path, *, progress=None):
    from mlx_lm import load
    import mlx.core as mx
    from mybot.services.history_training import tokenizer_hash
    root = adapter(workspace, version, require_quality=False)
    fingerprint = hashlib.sha256((root / 'adapters.safetensors').read_bytes()).hexdigest()
    lesson = StyleLesson(workspace)
    run = read_json(root / 'run.json')
    manifest = read_json(lesson.root / 'reviewed' / f"{run['dataset_id']}.json")
    cases = lesson.holdout_for(manifest.get('holdout_ref', 'holdout.json'))['cases']
    check_actions = any(item.get('source_kind') in {'historical_observed_reply_action', 'owner_confirmed_action'}
                        for item in manifest.get('examples', {}).get('train', []))
    path = quality_path(workspace, version)
    report = {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
              'version': version, 'format_version': FORMAT_VERSION, 'status': 'checking',
              'adapter_sha256': fingerprint, 'tokenizer_hash': tokenizer_hash(model_path),
              'cases': cases, 'seeds': list(SEEDS), 'results': [], 'action_results': [],
              'quality_assessed': False}
    # Retain all prior diagnostic evidence when retrying this same format.
    if path.exists():
        from uuid import uuid4
        write_json(root / 'quality_history' / (uuid4().hex + '.json'), read_json(path))
    write_json(path, report)
    started = time.monotonic()
    try:
        for side in ('baseline', 'candidate'):
            model, tokenizer = load(str(model_path), adapter_path=str(root) if side == 'candidate' else None)
            for key in cases:
                case = lesson.store.get(key)
                context, _ = lesson.store.source(case)
                for seed in SEEDS:
                    messages = example(context, [])['prompt']
                    _, attempts = sample(model, tokenizer, messages, seed, count=1, attempts=1)
                    result = dict(attempts[0], side=side, case_id=key,
                                  context_message_ids=case['context_ids'], source_digest=case['source_digest'],
                                  adapter_sha256=fingerprint if side == 'candidate' else None)
                    report['results'].append(result)
                    write_json(path, report)
                    total = len(cases) * len(SEEDS) * (3 if check_actions else 2)
                    if side == 'candidate' and check_actions:
                        from mybot.ai.action_scoring import score_actions
                        try:
                            action, scores = score_actions(model, tokenizer, decision_messages(context))
                        except ValueError as error:
                            action, scores = None, {'error': str(error)[:160]}
                        report['action_results'].append({'case_id': key, 'seed': seed,
                            'action': action, 'scores': scores,
                            'adapter_sha256': fingerprint})
                        write_json(path, report)
                    if progress:
                        progress(len(report['results']) + len(report['action_results']), total)
            del model, tokenizer
            mx.clear_cache()
        # Validate sources and the file again after GPU work.
        adapter(workspace, version, require_quality=False)
        if fingerprint != hashlib.sha256((root / 'adapters.safetensors').read_bytes()).hexdigest():
            raise ValueError('Адаптер изменился во время проверки.')
        report['counts'] = {side: sum(x['side'] == side and x['rejection'] is None for x in report['results'])
                            for side in ('baseline', 'candidate')}
        report['expected_per_model'] = len(cases) * len(SEEDS)
        report['expected_actions'] = report['expected_per_model'] if check_actions else 0
        report['valid_actions'] = sum(bool(item['action']) for item in report['action_results'])
        report['status'] = ('passed' if report['counts']['candidate'] == report['expected_per_model']
                            and report['valid_actions'] == report['expected_actions'] else 'failed')
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(path, report)
        if report['status'] == 'passed':
            activate_practice(workspace, version)
    except BaseException as error:
        report.update(status='interrupted_or_failed', error=type(error).__name__)
        write_json(path, report)
        raise
    return report

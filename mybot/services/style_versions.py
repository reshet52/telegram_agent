"""Resolve and explicitly activate same-dialog local adapters."""

import hashlib
import re

from mybot.storage.atomic import read_json, write_json
from mybot.storage.style_lesson import StyleLesson
from mybot.ai.local_format import FORMAT_VERSION


def quality_path(workspace, version):
    if not re.fullmatch(r'[a-f0-9]{32}', version):
        raise ValueError('Некорректная версия модели.')
    return workspace.root / 'learning/mlx_runs' / version / f'quality_{FORMAT_VERSION}.json'


def quality_passed(workspace, version):
    report = read_json(quality_path(workspace, version), {})
    run = read_json(quality_path(workspace, version).parent / 'run.json', {})
    if run.get('base_model'):
        from mybot.services.history_training import tokenizer_hash
        if report.get('tokenizer_hash') != tokenizer_hash(run['base_model']):
            return False
    lesson = StyleLesson(workspace)
    manifest = read_json(workspace.root / 'learning/reviewed' / (run.get('dataset_id', '') + '.json'), {})
    holdout = lesson.holdout_for(manifest.get('holdout_ref', 'holdout.json'))
    checks_actions = any(item.get('source_kind') in {'historical_observed_reply_action', 'owner_confirmed_action'}
                         for item in manifest.get('examples', {}).get('train', []))
    if manifest.get('version') == 4:
        import math
        return bool(report.get('status') == 'passed'
                    and report.get('method') == 'online_validation_and_instructions_v1'
                    and report.get('format_version') == FORMAT_VERSION
                    and (report.get('account_id'), report.get('dialog_id')) == (workspace.account_id, workspace.dialog_id)
                    and report.get('adapter_sha256') == run.get('adapter_sha256')
                    and report.get('cases') == (holdout or {}).get('cases', [])[:2]
                    and report.get('finite_weights') and report.get('validation_passed')
                    and len(report.get('results', [])) == 4
                    and sum(bool(r.get('instruction')) for r in report['results']) == 2
                    and all(r.get('passed') for r in report['results'])
                    and math.isfinite(run.get('loss_after', float('nan')))
                    and run['loss_after'] <= run['loss_before']
                    and run.get('completed_steps') == run.get('steps'))
    if manifest.get('version') == 3:
        return bool(report.get('status') == 'passed' and report.get('format_version') == FORMAT_VERSION
                    and (report.get('account_id'), report.get('dialog_id')) == (workspace.account_id, workspace.dialog_id)
                    and report.get('adapter_sha256') == run.get('adapter_sha256')
                    and report.get('cases') == (holdout or {}).get('cases')
                    and report.get('method') == '20_independent_cases_one_seed_current_vs_candidate'
                    and report.get('expected_per_model') == 20
                    and report.get('counts', {}).get('candidate') == 20
                    and len(report.get('results', [])) == 20
                    and all(r.get('candidate', {}).get('rejection', 'missing') is None
                            for r in report['results']))
    return bool(report.get('status') == 'passed' and report.get('format_version') == FORMAT_VERSION
                and (report.get('account_id'), report.get('dialog_id')) == (workspace.account_id, workspace.dialog_id)
                and report.get('adapter_sha256') == run.get('adapter_sha256')
                and report.get('cases') == (holdout or {}).get('cases')
                and report.get('counts', {}).get('candidate') == 2 * len((holdout or {}).get('cases', []))
                and (not checks_actions or report.get('valid_actions') == report.get('expected_actions')
                     == 2 * len((holdout or {}).get('cases', [])))
                and report.get('expected_per_model', 0) >= 4)


def selected(workspace):
    state = StyleLesson(workspace).read('selected_model.json')
    return state['current'] if state else 'configured'


def practice_state(workspace):
    return StyleLesson(workspace).read('practice_model.json') or {}


def practice_selected(workspace):
    version = practice_state(workspace).get('current', 'base')
    return version if version == 'base' or quality_passed(workspace, version) else 'base'


def practice_label(workspace, version=None):
    explicit = version is not None
    pending = version is None and practice_state(workspace).get('current', 'base') != practice_selected(workspace)
    version = version or practice_selected(workspace)
    if version == 'base':
        if not explicit:
            return 'недоступны · адаптер не прошёл автопроверку' if pending else 'сначала обучите модель на истории'
        return 'Базовая модель · сохранённый старый ответ'
    run = read_json(workspace.root / 'learning/mlx_runs' / version / 'run.json', {})
    title = {'history': 'Обучена на истории', 'refine': 'Доработана по вашим оценкам'}.get(run.get('purpose'), 'Локальная модель')
    return title + ' · ' + version[:8]


def activate_practice(workspace, version):
    root = adapter(workspace, version)
    run = read_json(root / 'run.json')
    state = practice_state(workspace)
    bootstrap = version if run.get('purpose') == 'history' else run.get('parent_adapter')
    if run.get('purpose') == 'reviewed':
        bootstrap = state.get('bootstrap')
        if not bootstrap:
            # Legacy reviewed training is usable, but is not a history bootstrap.
            StyleLesson(workspace).write('practice_model.json', {
                'current': version, 'previous': state.get('current', 'base')})
            return
    if not bootstrap:
        raise ValueError('У версии отсутствует первоначальный адаптер.')
    if state.get('current') == version and state.get('bootstrap') == bootstrap:
        return
    StyleLesson(workspace).write('practice_model.json', {
        'current': version, 'previous': state.get('current', 'base'), 'bootstrap': bootstrap})


def used_in_history(workspace, case):
    version = practice_state(workspace).get('bootstrap')
    if not version:
        return False
    run = read_json(workspace.root / 'learning/mlx_runs' / version / 'run.json', {})
    manifest = read_json(workspace.root / 'learning/reviewed' / (run.get('dataset_id', '') + '.json'), {})
    return any(set(case['expected_ids']).intersection(item.get('response_ids', []))
               for item in manifest.get('examples', {}).get('train', []))


def label(version):
    return {'configured': 'Прежняя модель', 'base': 'Локальная · базовая'}.get(
        version, 'Локальная · ' + version[:8])


def adapter(workspace, version, *, require_quality=True):
    if version == 'base':
        return None
    if not re.fullmatch(r'[a-f0-9]{32}', version):
        raise ValueError('Некорректная версия модели.')
    root = workspace.root / 'learning' / 'mlx_runs' / version
    run = read_json(root / 'run.json')
    if (not run or run.get('status') != 'trained_unapproved'
            or (run.get('account_id'), run.get('dialog_id')) != (workspace.account_id, workspace.dialog_id)
            or not (root / 'adapters.safetensors').is_file()):
        raise ValueError('Адаптер не готов или относится к другому диалогу.')
    from mybot.services.reviewed_dataset import load_reviewed
    load_reviewed(workspace, run['dataset_id'])
    parent = run.get('parent_adapter')
    if parent:
        if run.get('purpose') not in ('refine', 'online') or parent == version or not re.fullmatch(r'[a-f0-9]{32}', parent):
            raise ValueError('Некорректная цепочка адаптеров.')
        parent_run = read_json(workspace.root / 'learning/mlx_runs' / parent / 'run.json', {})
        if run.get('purpose') != 'online' and (parent_run.get('purpose') != 'history' or parent_run.get('parent_adapter')):
            raise ValueError('Нет первоначального адаптера.')
        adapter(workspace, parent, require_quality=False)
        if run.get('parent_sha256') != parent_run.get('adapter_sha256'):
            raise ValueError('Первоначальный адаптер изменился.')
    actual = hashlib.sha256((root / 'adapters.safetensors').read_bytes()).hexdigest()
    if run.get('adapter_sha256') != actual:
        raise ValueError('Файл адаптера изменён или не прошёл проверку целостности.')
    if require_quality and not quality_passed(workspace, version):
        raise ValueError('Адаптер требует автопроверки генерации. Откройте «Обучение и версии».')
    return root


def activate(workspace, version, *, explicit_owner=False, automatic=False):
    if version not in {'configured', 'base'}:
        adapter(workspace, version)
        comparison = StyleLesson(workspace).read(f'comparisons/{version}_{FORMAT_VERSION}.json')
        if automatic:
            run = read_json(adapter(workspace, version) / 'run.json')
            if run.get('purpose') != 'online' or run.get('parent_adapter') != selected(workspace):
                raise ValueError('Кандидат не продолжает действующую версию.')
        if not explicit_owner and not automatic and (not comparison or len(comparison['votes']) != len(comparison['cases'])):
            raise ValueError('Сначала завершите сравнение версий.')
    old = selected(workspace)
    if old != version:
        StyleLesson(workspace).write('selected_model.json', {'current': version, 'previous': old})


def rollback(workspace):
    state = StyleLesson(workspace).read('selected_model.json')
    if not state or not state.get('previous'):
        raise ValueError('Предыдущей версии нет.')
    activate(workspace, state['previous'], explicit_owner=True)

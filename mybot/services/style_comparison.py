"""Cached blind comparisons on the fixed holdout, never training targets."""

import random
from copy import deepcopy

from mybot.ai import local_reply
from mybot.services.learning_dataset import example
from mybot.services.style_versions import adapter, quality_path
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.atomic import read_json
from mybot.ai.local_format import FORMAT_VERSION


def comparison_name(version):
    return f'comparisons/{version}_{FORMAT_VERSION}.json'


def open_comparison(workspace, version, baseline='base'):
    root = adapter(workspace, version)
    lesson = StyleLesson(workspace)
    run = read_json(root / 'run.json')
    manifest = read_json(lesson.root / 'reviewed' / f"{run['dataset_id']}.json")
    holdout = lesson.holdout_for(manifest.get('holdout_ref', 'holdout.json'))
    name = comparison_name(version)
    data = lesson.read(name)
    if data is None:
        night = read_json(root / 'comparison.json', {})
        if manifest.get('version') == 3 and night.get('status') == 'passed':
            baseline = night['baseline']
            if baseline == 'configured':
                baseline = 'base'
        if baseline == version:
            baseline = 'base'
        data = {'candidate': version, 'baseline': baseline, 'format_version': FORMAT_VERSION, 'cases': holdout['cases'],
                'pairs': {}, 'votes': {},
                'orders': {str(i): random.sample(['baseline', 'candidate'], 2)
                           for i in range(len(holdout['cases']))}}
        # Reuse the real, matched-seed gate outputs instead of paying for a second
        # stochastic draw. Legacy comparisons remain in their original file.
        report = read_json(quality_path(workspace, version), {})
        prototypes = {}
        if manifest.get('version') == 3 and night.get('status') == 'passed':
            for index, key in enumerate(data['cases']):
                result = next(r for r in night['results'] if r['case_id'] == key)
                context = result['context']
                for side, source in (('baseline', 'current'), ('candidate', 'candidate')):
                    item = result[source]
                    request = local_reply.snapshot('', workspace, data[side],
                                                   messages=example(context, [])['prompt'], count=1)
                    request.update(seed=night['seed'], context_message_ids=[m['message_id'] for m in context],
                                   receipt={'seconds': item['seconds'],
                                            'adapter_sha256': night['baseline_sha256'] if side == 'baseline' else night['candidate_sha256'],
                                            'format_version': FORMAT_VERSION, 'source': 'night_comparison'})
                    data['pairs'].setdefault(str(index), {})[side] = {'text': item['text'], 'request': request}
        elif baseline == 'base':
            for index, key in enumerate(data['cases']):
                context = None
                for side in ('baseline', 'candidate'):
                    matches = [r for r in report.get('results', []) if r['case_id'] == key
                               and r['side'] == side and r['seed'] == 42 and not r['rejection']]
                    if not matches:
                        continue
                    result = matches[0]
                    if context is None:
                        context, _ = lesson.store.source(lesson.store.get(key))
                    messages = example(context, [])['prompt']
                    if side not in prototypes:
                        prototypes[side] = local_reply.snapshot('', workspace, data[side], messages=messages, count=1)
                    request = deepcopy(prototypes[side])
                    request['messages'] = messages
                    request.update(seed=42, context_message_ids=result['context_message_ids'],
                                   receipt={'seconds': result['seconds'], 'adapter_sha256': result['adapter_sha256'],
                                            'format_version': FORMAT_VERSION, 'source': 'generation_check'})
                    data['pairs'].setdefault(str(index), {})[side] = {'text': result['text'], 'request': request}
        lesson.write(name, data)
    if data['cases'] != holdout['cases']:
        raise ValueError('Проверочные отрывки изменились. Сравнение недоступно.')
    return data


async def generate_pair(workspace, version, index):
    lesson = StyleLesson(workspace)
    data = open_comparison(workspace, version)
    if not 0 <= index < len(data['cases']):
        raise ValueError('Нет такого проверочного отрывка.')
    case = lesson.store.get(data['cases'][index])
    context, _ = lesson.store.source(case)
    pair = data['pairs'].setdefault(str(index), {})
    for side in ('baseline', 'candidate'):
        if side in pair:
            continue
        request = local_reply.snapshot('', workspace, data[side], messages=example(context, [])['prompt'], count=1)
        request['seed'] = 42  # Same seed for both versions, cached per format.
        request['context_message_ids'] = case['context_ids']
        answer = (await local_reply.generate(request))[0]
        pair[side] = {'text': answer, 'request': request}
        lesson.write(comparison_name(version), data)
    return data


def vote(workspace, version, index, choice):
    lesson = StyleLesson(workspace)
    data = open_comparison(workspace, version)
    key = str(index)
    if key in data['votes']:
        return data
    if choice not in {'first', 'second', 'both', 'neither'} or len(data['pairs'].get(key, {})) != 2:
        raise ValueError('Сначала дождитесь обоих ответов.')
    data['votes'][key] = data['orders'][key][int(choice == 'second')] if choice in {'first', 'second'} else choice
    lesson.write(comparison_name(version), data)
    return data


def flag(workspace, version, index, position, issue):
    data = open_comparison(workspace, version)
    key = str(index)
    if issue not in {'reaction', 'facts', 'emoji', 'wording'} or position not in {0, 1} or len(data['pairs'].get(key, {})) != 2:
        raise ValueError('Некорректная оценка сравнения.')
    side = data['orders'][key][position]
    flags = data.setdefault('issues', {}).setdefault(key, {}).setdefault(side, [])
    if issue in flags:
        flags.remove(issue)
    else:
        flags.append(issue)
    StyleLesson(workspace).write(comparison_name(version), data)
    return data


def issue_summary(data):
    labels = {'reaction': 'Не та реакция', 'facts': 'Выдуманные факты',
              'emoji': 'Чужие эмодзи', 'wording': 'Чужие формулировки'}
    lines = []
    for issue, title in labels.items():
        counts = [sum(issue in case.get(side, []) for case in data.get('issues', {}).values())
                  for side in ('baseline', 'candidate')]
        lines.append(f'{title}: {counts[0]} / {counts[1]}')
    return '\n'.join(lines)

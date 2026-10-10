"""Frozen, source-disjoint owner evaluation; never infer owner style ratings."""

import json
import re
import fcntl
from contextlib import contextmanager
from uuid import uuid4

from mybot.services.conversation_boundaries import gap_seconds
from mybot.services.full_history_dataset import previous_training_ids
from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, candidates, now
from mybot.storage.style_lesson import StyleLesson


def path(ws):
    return ws.root / 'reply_assistant/evaluation.json'


def load(ws):
    value = read_json(path(ws))
    if value and (value.get('account_id'), value.get('dialog_id')) != (ws.account_id, ws.dialog_id):
        raise ValueError('Проверка принадлежит другому диалогу.')
    return value


@contextmanager
def state_lock(ws):
    root = ws.root / 'reply_assistant'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'evaluation_state.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def save_progress(ws, value):
    """A running worker must never overwrite concurrently imported owner ratings."""
    with state_lock(ws):
        current = load(ws)
        if not current or current['id'] != value['id']:
            raise ValueError('Проверка изменилась во время выполнения.')
        ratings = {case['id']: case['ratings'] for case in current['cases']}
        for case in value['cases']:
            case['ratings'] = ratings[case['id']]
        write_json(path(ws), value)


def reserved(ws):
    value = load(ws)
    return {i for item in (value or {}).get('cases', []) for i in item['source_ids']}


def tags(text):
    result = []
    if '?' in text:
        result.append('question')
    patterns = {'greeting': r'\b(hey|hi|hello|привет|доброе)\b',
                'plans': r'\b(tomorrow|tonight|when|time|meet|завтра|сегодня|встреч|когда)',
                'emotional': r'love|miss|sorry|sad|❤️|люб|скуч|груст',
                'refusal': r'\b(no|not|cannot|can.t|нет|не хочу|не могу)\b'}
    result += [name for name, pattern in patterns.items() if re.search(pattern, text.casefold())]
    return result or ['ambiguous']


def prepare(ws, count=50):
    existing = load(ws)
    if existing:
        validate(ws, existing)
        return existing
    store = PracticeStore(ws)
    history = store.history()
    used = previous_training_ids(StyleLesson(ws))
    pending = read_json(ws.root / 'learning/online_learning.json', {})
    if pending and (pending.get('account_id'), pending.get('dialog_id')) != (ws.account_id, ws.dialog_id):
        raise ValueError('Очередь принадлежит другому диалогу.')
    used.update(i for item in pending.get('queue', []) for i in item['context_ids'] + item['response_ids'])
    for p in (ws.root / 'learning/confirmed').glob('*.json'):
        review = read_json(p, {})
        used.update(review.get('context_ids', []) + review.get('response_ids', []))
    # Exclude even older exported training sets whose run did not leave a manifest.
    for p in (ws.root / 'learning/datasets').glob('*/sources.json'):
        used.update(i for item in read_json(p, {}).get('train', [])
                    for i in item['context_ids'] + item['response_ids'])
    for p in (ws.root / 'learning/reviewed').glob('*.json'):
        used.update(i for item in read_json(p, {}).get('examples', {}).get('train', []) for i in item['ids'])
    chosen = []
    for context, expected in reversed(candidates(history, load_deleted_message_ids(ws.deleted_message_ids),
                                                 set(), threshold=gap_seconds(ws))):
        # Short independent windows, but never trim the final incoming group.
        start = len(context) - 1
        while start > 0 and context[start - 1]['sender'] != 'Я':
            start -= 1
        context = context[min(start, max(0, len(context) - 6)):]
        ids = [m['message_id'] for m in context + expected]
        if set(ids) & used:
            continue
        chosen.append((context, expected))
        used.update(ids)
        if len(chosen) == count:
            break
    cases = []
    for index, (context, expected) in enumerate(reversed(chosen)):
        case = store.create_from(context, expected, 'reply_evaluation')
        categories = tags(context[-1]['text'])
        instruction = 'Ответь естественно, полностью сохраняя мой стиль общения.'
        if index % 5 == 4:
            instruction += ' Ответь коротко, без эмодзи.'
            categories.append('explicit_instruction')
        cases.append({'id': case['id'], 'source_ids': case['context_ids'] + case['expected_ids'],
                      'source_digest': case['source_digest'], 'categories': categories,
                      'instruction': instruction, 'results': {}, 'ratings': {}})
    value = {'id': uuid4().hex, 'account_id': ws.account_id, 'dialog_id': ws.dialog_id,
             'created_at': now(), 'requested_cases': count, 'cases': cases,
             'status': 'prepared' if len(cases) == count else 'insufficient_independent_sources',
             'baseline': None, 'pipeline_version': 1}
    write_json(path(ws), value)
    return value


def validate(ws, value):
    if (value.get('account_id'), value.get('dialog_id')) != (ws.account_id, ws.dialog_id):
        raise ValueError('Чужая проверка.')
    trained = previous_training_ids(StyleLesson(ws))
    used = set()
    store = PracticeStore(ws)
    history = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(ws.deleted_message_ids)
    for item in value['cases']:
        case = store.get(item['id'])
        store.source(case, messages=history, deleted=deleted)
        if case['source_digest'] != item['source_digest'] or set(item['source_ids']) & (used | trained):
            raise ValueError('Контрольные источники изменены, пересекаются или использованы в обучении.')
        used.update(item['source_ids'])


def import_ratings(ws, filename):
    with state_lock(ws):
        _import_ratings(ws, filename)
    return report(ws)


def _import_ratings(ws, filename):
    data, incoming = load(ws), read_json(filename)
    validate(ws, data)
    if not incoming or incoming.get('evaluation_id') != data['id']:
        raise ValueError('Оценки относятся к другой проверке.')
    known = {item['id']: item for item in data['cases']}
    for key, ratings in incoming.get('ratings', {}).items():
        if key not in known or not isinstance(ratings, dict):
            raise ValueError('Неизвестная ситуация в оценках.')
        item = known[key]
        for mode, by_number in ratings.items():
            if mode not in item['results'] or not isinstance(by_number, dict):
                raise ValueError('Неизвестный результат в оценках.')
            numbers = {str(v['number']) for v in item['results'][mode]['variants']}
            for number, rating in by_number.items():
                if (number not in numbers or not isinstance(rating, dict)
                        or rating.get('rating') not in {'ready', 'edit', 'bad'}
                        or any(type(rating.get(k)) is not bool for k in ('meaning_error', 'contradiction'))):
                    raise ValueError('Некорректная оценка.')
        item['ratings'] = ratings
    write_json(path(ws), data)


def report(ws):
    value = load(ws)
    store = PracticeStore(ws)
    summary = {'cases': len(value['cases']), 'categories': {}, 'modes': {}}
    for case in value['cases']:
        for category in case['categories']:
            summary['categories'][category] = summary['categories'].get(category, 0) + 1
    for mode in ('baseline', 'enhanced'):
        complete, ready, contradictions, meaning_errors = 0, 0, 0, 0
        for case in value['cases']:
            result = case['results'].get(mode)
            rated = case['ratings'].get(mode, {})
            numbers = {str(v['number']) for v in (result or {}).get('variants', [])}
            if result and (not numbers or numbers <= rated.keys()):
                complete += 1
                ready += any(v['rating'] == 'ready' and not v['meaning_error'] and not v['contradiction'] for v in rated.values())
            contradictions += sum(v['contradiction'] for v in rated.values())
            meaning_errors += sum(v['meaning_error'] for v in rated.values())
        summary['modes'][mode] = {'fully_rated': complete, 'ready_cases': ready,
            'generated': sum(mode in c['results'] for c in value['cases']),
            'ready_fraction': ready / complete if complete else None,
            'contradictions': contradictions, 'meaning_errors': meaning_errors}
    summary['missing_categories'] = sorted({'question', 'greeting', 'plans', 'emotional',
        'ambiguous', 'refusal', 'explicit_instruction'} - summary['categories'].keys())
    summary['target_met'] = (len(value['cases']) == value['requested_cases'] == 50
        and summary['modes']['enhanced']['fully_rated'] == 50
        and summary['modes']['enhanced']['ready_fraction'] >= .8
        and summary['modes']['enhanced']['contradictions'] == 0)
    write_json(ws.root / 'reply_assistant/evaluation_summary.json', summary)
    entries = []
    for case in value['cases']:
        context, _ = store.source(store.get(case['id']))
        entries.append(dict(case, context=context))
    # Standalone local review page, no external assets or network requests.
    payload = json.dumps({'evaluation_id': value['id'], 'cases': entries}, ensure_ascii=False).replace('<', '\u003c')
    page = r'''<!doctype html><meta charset="utf-8"><title>Проверка ответов</title>
<style>body{font:17px system-ui;max-width:1000px;margin:30px auto;background:#17222c;color:#eee}
article,section{padding:18px;margin:15px 0;background:#243543;border-radius:12px}
pre{white-space:pre-wrap;font:inherit}button,select{font:inherit;margin:8px;padding:8px}
label{display:block;margin:8px}</style>
<h1>Ответы без правок</h1><p>Оцените каждый готовый вариант. Проверка не добавляет ответы в обучение.</p>
<button id="save">Сохранить оценки</button><div id="root"></div><script>
const data=PAYLOAD, ratings={};
const storageKey='mybot-reply-evaluation-'+data.evaluation_id;let saved={};
try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}');}catch(e){}
const node=(tag,text)=>{const n=document.createElement(tag);if(text)n.textContent=text;return n};
document.getElementById('root').append(node('p','Подготовлено полных сравнений: '+data.cases.filter(c=>c.results.baseline&&c.results.enhanced).length+'/'+data.cases.length+'. Новые результаты появятся после обновления страницы.'));
for(const c of data.cases){ratings[c.id]=saved[c.id]||c.ratings||{};const a=node('article');
a.append(node('h2','Ситуация '+(data.cases.indexOf(c)+1)),node('pre',c.context.map(m=>m.sender+': '+m.text).join('\n\n')),
node('p','Указание: '+c.instruction));
for(const mode of ['baseline','enhanced']){const s=node('section');s.append(node('h3',mode==='baseline'?'Прежний подход':'Новый подход'));
const r=c.results[mode];if(!r){s.append(node('p','Ещё не выполнено'));a.append(s);continue;}
if(!r.variants.length)s.append(node('p','Готовых ответов нет'));
if(r.preparation?.status==='needs_clarification')s.append(node('p','Уточнение: '+r.preparation.result.question));
for(const v of r.variants){s.append(node('pre','Вариант '+v.number+' · '+v.source+'\n'+v.text));
ratings[c.id][mode]??={};const old=ratings[c.id][mode][v.number];
const select=node('select');for(const [key,title] of [['','Выберите оценку'],['ready','Отправлю без правок'],['edit','Нужна правка'],['bad','Не подходит']]){
const option=node('option',title);option.value=key;select.append(option);}select.value=old?.rating||'';s.append(select);
const boxes={};for(const [key,title] of [['meaning_error','Ошибка смысла'],['contradiction','Противоречие фактам']]){
const label=node('label'),box=node('input');box.type='checkbox';box.checked=old?.[key]||false;boxes[key]=box;label.append(box,document.createTextNode(title));s.append(label);}
const update=()=>{if(!select.value){delete ratings[c.id][mode][v.number];return;}ratings[c.id][mode][v.number]={rating:select.value,
meaning_error:boxes.meaning_error.checked,contradiction:boxes.contradiction.checked};
try{localStorage.setItem(storageKey,JSON.stringify(ratings));}catch(e){}};select.onchange=update;for(const b of Object.values(boxes))b.onchange=update;
}a.append(s);}document.getElementById('root').append(a);}
document.getElementById('save').onclick=()=>{const a=node('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({evaluation_id:data.evaluation_id,ratings},null,2)],{type:'application/json'}));
a.download='reply-ratings.json';a.click();URL.revokeObjectURL(a.href);};</script>'''.replace('PAYLOAD', payload)
    write_text(ws.root / 'reply_assistant/evaluation.html', page)
    text = '# Проверка ответов\n\n' + '\n'.join(
        f"{mode}: выполнено {s['generated']}/{summary['cases']}, оценено {s['fully_rated']}/{summary['cases']}; "
        + (f"готовый ответ {s['ready_cases']}/{s['fully_rated']}; " if s['fully_rated'] else 'пригодность без правок ещё не оценена; ')
        + f"ошибок смысла {s['meaning_errors']}, противоречий {s['contradictions']}." for mode, s in summary['modes'].items())
    text += '\n\nЦель 80%: ' + ('достигнута.' if summary['target_met'] else 'пока не подтверждена.')
    text += '\n\nКатегории: ' + json.dumps(summary['categories'], ensure_ascii=False)
    if summary['missing_categories']:
        text += '\n\nВ независимой истории не найдены категории: ' + ', '.join(summary['missing_categories'])
    write_text(ws.root / 'reply_assistant/report.md', text)
    return summary

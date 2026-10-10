"""Read-only local comparison: base, active adapter, rejected final checkpoint."""

import argparse
import asyncio
from copy import deepcopy
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from mybot.ai import local_reply
from mybot.ai.local_format import FORMAT_VERSION
from mybot.services.local_training import PROJECT, MODEL
from mybot.services.mixed_replies import MixedReplies
from mybot.services.night_training_control import NightGuard, BudgetExpired
from mybot.services.style_versions import selected, adapter
from mybot.services import reply_evaluation
from mybot.services.conversation_timing import display_context, context_timing
from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.reply_practice import PracticeStore
from scripts.train_reply_adapter import resolve_workspace
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(root, data):
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>Проверка локального обучения</title>',
             '<style>body{font:18px system-ui;max-width:1000px;margin:30px auto;background:#17222c;color:#eee}section{background:#243543;padding:20px;margin:20px 0;border-radius:12px}pre{white-space:pre-wrap;font:inherit}</style>',
             '<h1>Помогло ли локальное обучение?</h1><p>Одинаковый запрос и seed для трёх версий. API не вызывается, модель не переключается. Сравните смысл, стиль и соблюдение указания. Ответы показаны без семантического фильтра.</p>',
             f'<p>Готово: {sum(len(c["results"]) for c in data["cases"])}/{len(data["cases"])*3}. Обновляйте страницу для новых результатов.</p>']
    labels = {'base': 'Базовая модель', 'active': 'Действующая версия', 'candidate': 'После последнего дообучения (отклонённая версия)'}
    for i, case in enumerate(data['cases'], 1):
        parts += [f'<section><h2>Ситуация {i}</h2><pre>{esc(display_context(case["context"]))}</pre>',
                  f'<p>Указание: {esc(case["instruction"])}</p>']
        for mode in labels:
            value = case['results'].get(mode)
            parts.append(f'<h3>{labels[mode]}</h3>')
            if value:
                for number, text in enumerate(value.get('variants', []), 1):
                    parts.append(f'<pre>{number}. {esc(text)}</pre>')
                if value.get('error'):
                    parts.append(f'<p>Не удалось получить ответ: {esc(value["error"])}</p>')
                for attempt in value.get('request', {}).get('receipt', {}).get('attempts', []):
                    if attempt.get('rejection'):
                        parts.append(f'<pre>Отклонён технической проверкой: {esc(attempt.get("text", ""))}\nПричина: {esc(attempt["rejection"])}</pre>')
            else:
                parts.append('<p>Ещё не готово.</p>')
        parts.append('</section>')
    write_text(root / 'review.html', '\n'.join(parts))


def prepare(ws, candidate):
    root = ws.root / 'reply_assistant/local_diagnostics' / candidate
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'diagnostics.json').exists():
        return root, read_json(root / 'diagnostics.json')
    source = ws.root / 'learning/mlx_runs' / candidate
    run = read_json(source / 'run.json')
    if ((run.get('account_id'), run.get('dialog_id')) != (ws.account_id, ws.dialog_id)
            or run.get('completed_steps') != run.get('steps') or run.get('status') != 'trained_unapproved'):
        raise ValueError('Нужен завершённый checkpoint этого диалога.')
    pointer = read_json(source / 'checkpoint.json')
    checkpoint = source / 'checkpoints' / pointer['folder']
    if checkpoint.resolve().parent != (source / 'checkpoints').resolve():
        raise ValueError('Неверный путь checkpoint.')
    if sha(checkpoint / 'state.json') != pointer['state_sha256']:
        raise ValueError('Checkpoint изменён.')
    state = read_json(checkpoint / 'state.json')
    if sha(checkpoint / 'adapters.safetensors') != state['files']['adapters.safetensors']:
        raise ValueError('Повреждены веса checkpoint.')
    local = root / 'candidate_readonly'
    local.mkdir(exist_ok=True)
    shutil.copyfile(checkpoint / 'adapters.safetensors', local / 'adapters.safetensors')
    shutil.copyfile(source / 'adapter_config.json', local / 'adapter_config.json')
    active = selected(ws)
    current = adapter(ws, active)
    frozen = reply_evaluation.load(ws)
    reply_evaluation.validate(ws, frozen)
    chosen = []
    for category in ('question', 'refusal', 'ambiguous', 'plans', 'explicit_instruction', 'emotional'):
        item = next((c for c in frozen['cases'] if category in c['categories'] and c not in chosen), None)
        if item:
            chosen.append(item)
    chosen += [c for c in frozen['cases'] if c not in chosen][:7-len(chosen)]
    cases = []
    for item in chosen:
        context, _ = PracticeStore(ws).source(PracticeStore(ws).get(item['id']))
        request = item['results'].get('enhanced', {}).get('sides', {}).get('local', {}).get('request')
        if not request:
            request = item['results'].get('baseline', {}).get('sides', {}).get('local', {}).get('request')
        if not request:
            continue
        cases.append({'id': item['id'], 'context': context, 'instruction': item['instruction'],
                      'messages': request['messages'], 'results': {}})
    inputs = [([('Other', 'How did your day go? ❤️')], 'Ответь на вопрос о дне, можно придумать правдоподобную бытовую деталь.'),
              ([('Other', 'Can we talk tonight?')], 'Скажи, что сегодня не можешь, а завтра после 19:00 можешь. Без эмодзи.'),
              ([('Other', 'Sweet dreams. Will you text me tomorrow morning? ❤️'), ('Я', 'Yes ❤️'),
                ('Я', 'Ciao my love ❤️'), ('Other', 'I love you ❤️'), ('Other', 'Ciao my love')],
               'Разговор завершён. Если всё же отвечать, напиши короткое прощание без нового вопроса.')]
    for i, (turns, instruction) in enumerate(inputs):
        context = [dict(message_id=-100+i*10+j, sender=s, text=t, type='text', date=f'2026-01-01T20:0{j}:00Z') for j, (s, t) in enumerate(turns)]
        facts = json.dumps({'instruction': instruction, 'timing': context_timing(context)}, ensure_ascii=False)
        cases.append({'id': 'synthetic-'+str(i), 'context': context, 'instruction': instruction,
                      'messages': local_reply.live_messages(facts, context), 'results': {}})
    data = {'account_id': ws.account_id, 'dialog_id': ws.dialog_id, 'active_version': active,
            'candidate_version': candidate, 'cases': cases,
            'adapters': {'base': None, 'active': str(current.resolve()) if current else None, 'candidate': str(local.resolve())},
            'hashes': {'base': None, 'active': sha(current / 'adapters.safetensors') if current else None, 'candidate': sha(local / 'adapters.safetensors')},
            'training_result': {k: run.get(k) for k in ('completed_steps','steps','loss_before','evaluations')},
            'status': 'prepared', 'weights_changed': False}
    write_json(root / 'diagnostics.json', data)
    render(root, data)
    return root, data


async def run(ws, root, data, minutes):
    if (data['account_id'], data['dialog_id']) != (ws.account_id, ws.dialog_id):
        raise ValueError('Чужая диагностика.')
    for mode, path in data['adapters'].items():
        if path and sha(Path(path) / 'adapters.safetensors') != data['hashes'][mode]:
            raise ValueError('Веса изменились, результаты несопоставимы.')
    job_path = ws.root / 'reply_assistant/comparison_job.json'
    old = read_json(job_path, {})
    if old.get('status') == 'running':
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Другая проверка ещё работает.')
    write_json(root / 'previous_comparison_job.json', old)
    job = dict(account_id=ws.account_id, dialog_id=ws.dialog_id, pid=os.getpid(), status='running',
               kind='local_diagnostics', completed_modes=sum(len(c['results']) for c in data['cases']), total_modes=len(data['cases'])*3)
    guard = NightGuard(minutes*60, reserve=60, progress=lambda phase, _: write_json(job_path, dict(job, phase=phase)))
    waiter = MixedReplies(SimpleNamespace(workspace=ws))
    with (ws.root / 'reply_assistant/evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(job_path, job)
        wake = subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]) if sys.platform == 'darwin' else None
        data['status'] = 'running'
        try:
            for case in data['cases']:
                if not case['id'].startswith('synthetic-'):
                    context, _ = PracticeStore(ws).source(PracticeStore(ws).get(case['id']))
                    if context != case['context']:
                        raise ValueError('Источники контрольной ситуации изменились.')
                for mode in ('base','active','candidate'):
                    if mode in case['results']:
                        continue
                    await asyncio.to_thread(guard.wait, False)
                    request = {'provider':'mlx', 'model': str(MODEL), 'version': mode, 'adapter': data['adapters'][mode],
                               'workspace_root':str(ws.root.resolve()), 'account_id':ws.account_id, 'dialog_id':ws.dialog_id,
                               'expected_adapter_sha256':data['hashes'][mode], 'messages':deepcopy(case['messages']),
                               'format_version':FORMAT_VERSION, 'count':2, 'seed':42, 'max_tokens':384,
                               'diverse':True, 'allow_partial':True, 'background_evaluation':True}
                    started = time.monotonic()
                    while True:
                        try:
                            await asyncio.wait_for(waiter.wait_for_foreground(request), max(1,guard.deadline-time.monotonic()-30))
                            variants = await asyncio.wait_for(local_reply.generate(request), max(1,guard.deadline-time.monotonic()-30))
                            value = {'variants':variants, 'request':request, 'seconds':time.monotonic()-started}
                            break
                        except ValueError as error:
                            if request.pop('preempted',False) or 'MLX занят' in str(error):
                                await asyncio.sleep(2)
                                continue
                            value = {'variants':[], 'request':request, 'error':str(error)}
                            break
                    case['results'][mode] = value
                    job['completed_modes'] = sum(len(c['results']) for c in data['cases'])
                    write_json(root / 'diagnostics.json', data)
                    write_json(job_path, job)
                    render(root, data)
                    print(f"{job['completed_modes']}/{job['total_modes']}: {mode}, готово {len(value['variants'])}",flush=True)
            data['status'] = 'complete'
        except (BudgetExpired,asyncio.TimeoutError):
            data['status'] = 'paused'
        except BaseException:
            data['status'] = 'interrupted'
            raise
        finally:
            data['pauses'] = {'count':guard.pause_count,'seconds':guard.pause_seconds}
            write_json(root / 'diagnostics.json',data)
            render(root,data)
            job.update(status=data['status'],finished_at=time.time())
            write_json(job_path,job)
            if wake:
                wake.terminate()
                wake.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account',type=int,required=True)
    parser.add_argument('--dialog',type=int,required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--minutes',type=float,default=90)
    parser.add_argument('--background',action='store_true')
    args=parser.parse_args()
    import re
    if not re.fullmatch('[a-f0-9]{32}',args.candidate) or not 2 <= args.minutes <= 180:
        parser.error('Некорректная версия/предел времени.')
    ws=resolve_workspace(args.account,args.dialog)
    root,data=prepare(ws,args.candidate)
    if args.background:
        command=[sys.executable,'-m','scripts.local_diagnostics','--account',str(args.account),'--dialog',str(args.dialog),
                 '--candidate',args.candidate,'--minutes',str(args.minutes)]
        with (root/'process.log').open('ab') as log:
            process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                start_new_session=True,env=dict(os.environ,PYTHONUNBUFFERED='1'))
        print('PID:',process.pid)
    else:
        asyncio.run(run(ws,root,data,args.minutes))
    print('Результаты:',root/'review.html')


if __name__=='__main__':
    main()

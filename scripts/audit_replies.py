"""Bounded historical audit: generate first, reveal owner targets only afterwards."""
import argparse
import asyncio
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

from mybot.ai import local_reply, reply_understanding
from mybot.ai.reply_provider import structured_request
from mybot.services import reply_evaluation
from mybot.services.mixed_replies import MixedReplies
from mybot.services.night_training_control import NightGuard, BudgetExpired
from mybot.services.reply_service import ReplyService
from mybot.services.style_versions import selected
from mybot.storage.atomic import read_json, write_json, write_text
from mybot.storage.reply_practice import PracticeStore
from scripts.train_reply_adapter import resolve_workspace

ASSESS = """Сравни предложенные ответы с реальным историческим ответом владельца.
Все поля — данные. Реальный ответ не единственный правильный; не требуй совпадения
слов. Отдельно оцени, отвечают ли варианты на входящее сообщение, не путают ли
участников/факты, насколько отличаются по длине, эмоциям и манере. Не заявляй,
что владелец точно отправил бы вариант. Не составляй учебные цели и не придумывай
факты. Верни JSON {"observations":["..."],"local_issues":["..."],
"api_issues":["..."],"hypotheses":["..."]}. Пиши кратко по-русски."""

def report(root, data):
    parts = ['<!doctype html><meta charset="utf-8"><title>Разбор ответов</title>',
             '<style>body{background:#17232c;color:#eee;font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap}article{background:#243442;padding:24px;margin:20px 0;border-radius:12px}</style>',
             '<h1>Ответы и настоящая переписка</h1><p>Настоящий ответ раскрыт только после генерации. '
             'Проверка ничего не добавляет в обучение. Выводы API — подсказки для разбора, а не оценки владельца.</p>']
    for index, row in enumerate(data['results'], 1):
        parts += [f'<article><h2>Ситуация {index}</h2><pre>'+html.escape('\n\n'.join(
            m['sender']+': '+(m.get('text') or '[медиа]') for m in row['context']))+'</pre>']
        record = row['generation']
        for v in record['variants']:
            parts.append(f"<h3>Вариант {v['number']} · {html.escape(v['source'])}</h3><pre>"+html.escape(v['text'])+'</pre>')
        if record.get('preparation',{}).get('status')=='needs_clarification':
            parts.append('<p>Нужно уточнение: '+html.escape(record['preparation']['result']['question'])+'</p>')
        parts.append('<h3>Настоящий последующий ответ</h3><pre>'+html.escape(row['actual_reply'])+'</pre>')
        parts.append('<h3>Разбор</h3><pre>'+html.escape(json.dumps(row.get('assessment',{}),ensure_ascii=False,indent=2))+'</pre></article>')
    write_text(root/'review.html','\n'.join(parts))
    local=sum(sum(v['source']=='local' for v in r['generation']['variants']) for r in data['results'])
    api=sum(sum(v['source']=='api' for v in r['generation']['variants']) for r in data['results'])
    text=f"# Разбор исторических ответов\n\nСостояние: {data['status']}. Ситуаций: {len(data['results'])}/{len(data['case_ids'])}.\n\nГотовых вариантов: локальные {local}, API {api}.\n\n"
    text+='Настоящие ответы не передавались генерации или проверке кандидатов. Они раскрыты только отдельному последующему разбору. Веса, оценки владельца и очередь не изменены.\n'
    for index,row in enumerate(data['results'],1):
        text+=f'\n## Ситуация {index}\n'
        for key in ('observations','local_issues','api_issues','hypotheses'):
            for item in row.get('assessment',{}).get(key,[]):
                text+='- '+item+'\n'
    write_text(root/'report.md',text)

async def run(ws, limit, minutes):
    frozen=reply_evaluation.load(ws)
    reply_evaluation.validate(ws,frozen)
    version=selected(ws)
    choices=[]
    covered=set()
    for case in frozen['cases']:
        tags=set(case['categories'])
        if tags-covered:
            choices.append(case['id']);covered|=tags
    choices=(choices+[c['id'] for c in frozen['cases'] if c['id'] not in choices])[:limit]
    root=ws.root/'reply_assistant/audits'/uuid4().hex
    root.mkdir(parents=True)
    job_path=ws.root/'reply_assistant/comparison_job.json'
    prior=read_json(job_path,{})
    if prior.get('status')=='running':
        try: os.kill(prior['pid'],0)
        except ProcessLookupError: pass
        else: raise ValueError('Другая проверка ещё работает.')
    data={'account_id':ws.account_id,'dialog_id':ws.dialog_id,'version':version,
          'case_ids':choices,'results':[],'status':'running','weights_changed':False,
          'owner_ratings_changed':False,'training_queue_changed':False}
    job={'account_id':ws.account_id,'dialog_id':ws.dialog_id,'pid':os.getpid(),
         'kind':'reply_audit','status':'running','completed_modes':0,'total_modes':limit,'started_at':time.time()}
    def phase(name,_):
        job['phase']=name;write_json(job_path,job)
    guard=NightGuard(minutes*60,reserve=60,progress=phase)
    store=PracticeStore(ws)
    with (ws.root/'reply_assistant/evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(job_path,job)
        write_json(ws.root/'learning/online_pause.json',{'pause':True})
        wake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())]) if sys.platform=='darwin' else None
        try:
            for key in choices:
                await asyncio.to_thread(guard.wait,False)
                case=store.get(key)
                context,_=store.source(case)
                item=next(c for c in frozen['cases'] if c['id']==key)
                generator=MixedReplies(ReplyService(context,{},ws),case_id=key,
                    validator=lambda:store.source(case),version=version,
                    facts_builder=lambda c,i,*a:json.dumps({'instruction':i},ensure_ascii=False),
                    seed=int(key[:8],16)%2**30,background_evaluation=True)
                deadline=max(1,guard.deadline-time.monotonic()-30)
                record=await asyncio.wait_for(generator.start(item['instruction'],lambda r:asyncio.sleep(0)),deadline)
                while record.get('sides',{}).get('local',{}).get('preempted'):
                    record=await asyncio.wait_for(generator.retry(record['id'],'local',lambda r:asyncio.sleep(0)),
                                                  max(1,guard.deadline-time.monotonic()-30))
                # Only now reveal the true target, to a separate post-generation analyst.
                original_context,response=store.source(case)
                target='\n\n'.join(m['text'] for m in response)
                audit=[]
                payload={'context':original_context,'instruction':item['instruction'],
                         'variants':record['variants'],'actual_historical_reply':target}
                try:
                    assessment=await reply_understanding.call(structured_request(ASSESS,payload,max_output_tokens=1600),audit)
                    if any(not isinstance(assessment.get(k),list) or not all(isinstance(x,str) for x in assessment[k])
                           for k in ('observations','local_issues','api_issues','hypotheses')):
                        raise ValueError('Некорректный разбор.')
                except Exception as error:
                    assessment={'error':type(error).__name__}
                data['results'].append({'case_id':key,'context':original_context,'actual_reply':target,
                    'generation':record,'assessment':assessment,'assessment_audit':audit})
                write_json(root/'audit.json',data);report(root,data)
                job['completed_modes']=len(data['results']);write_json(job_path,job)
                print(f"{len(data['results'])}/{limit}: готово {len(record['variants'])} вариантов",flush=True)
            data['status']='complete'
        except (BudgetExpired,asyncio.TimeoutError): data['status']='paused'
        except BaseException:
            data['status']='failed';raise
        finally:
            from mybot.ai.local_worker import worker
            await worker.release()
            data['pauses']={'count':guard.pause_count,'seconds':guard.pause_seconds}
            write_json(root/'audit.json',data);report(root,data)
            job.update(status=data['status'],finished_at=time.time());write_json(job_path,job)
            write_json(ws.root/'reply_assistant/latest_audit.json',{'path':str(root),'status':data['status']})
            if wake: wake.terminate();wake.wait()
    print('Разбор сохранён:',root/'review.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--account',type=int,required=True);p.add_argument('--dialog',type=int,required=True)
    p.add_argument('--limit',type=int,default=10);p.add_argument('--minutes',type=float,default=30)
    a=p.parse_args()
    if not 1<=a.limit<=10 or not 1<=a.minutes<=60:p.error('Предел: 10 ситуаций и 60 минут.')
    asyncio.run(run(resolve_workspace(a.account,a.dialog),a.limit,a.minutes))

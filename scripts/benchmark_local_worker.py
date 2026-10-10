"""Five matched local-only cold/resident measurements; never trains or activates."""
import argparse
import asyncio
from copy import deepcopy
import fcntl
import json
import os
import hashlib
import subprocess
import sys
import time
from statistics import median
from mybot.ai import local_reply
from mybot.ai.local_worker import worker
from mybot.services.local_training import PROJECT, PYTHON
from mybot.services.style_versions import selected
from mybot.services import reply_evaluation
from mybot.services.mixed_replies import MixedReplies
from mybot.services.reply_service import ReplyService
from mybot.services.night_training_control import NightGuard, BudgetExpired
from mybot.storage.atomic import read_json, write_json, write_text
from scripts.train_reply_adapter import resolve_workspace

async def run(ws,seconds=900,*,prefix=False):
    frozen=reply_evaluation.load(ws)
    reply_evaluation.validate(ws,frozen)
    inputs=[]
    for item in frozen['cases']:
        request=item['results'].get('enhanced',{}).get('sides',{}).get('local',{}).get('request')
        if request:
            current=local_reply.snapshot('',ws,selected(ws),messages=request['messages'],count=2)
            current.update(seed=42,diverse=True,allow_partial=True,background_evaluation=True)
            inputs.append((item['id'],current))
        if len(inputs)==5:
            break
    if len(inputs)!=5:
        raise ValueError('Нужно пять готовых защищённых ситуаций.')
    modes=('plain','prefix') if prefix else ('cold','resident')
    stem='prefix_benchmark' if prefix else 'worker_benchmark'
    root=ws.root/'reply_assistant'
    job_path=root/'comparison_job.json'
    old=read_json(job_path,{})
    if old.get('status')=='running':
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('Другая проверка ещё работает.')
    write_json(root/(stem+'_previous_job.json'),old)
    data=dict(account_id=ws.account_id,dialog_id=ws.dialog_id,version=selected(ws),results=[],weights_changed=False,status='running')
    data['input_sha256']=hashlib.sha256(json.dumps(inputs,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    previous=read_json(root/(stem+'.json'),{})
    if (previous.get('version')==data['version'] and previous.get('account_id')==ws.account_id
            and previous.get('dialog_id')==ws.dialog_id and previous.get('status')!='complete'
            and previous.get('input_sha256')==data['input_sha256']):
        data['results']=previous.get('results',[])
    job=dict(account_id=ws.account_id,dialog_id=ws.dialog_id,pid=os.getpid(),status='running',kind='worker_benchmark',completed_modes=len(data['results']),total_modes=10)
    def phase(name,thermal):
        job.update(phase=name,thermal=thermal)
        write_json(job_path,job)
    guard=NightGuard(seconds,reserve=60,progress=phase)
    priority=MixedReplies(ReplyService([],{},ws),background_evaluation=True)
    with (root/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(job_path,job)
        write_json(root/(stem+'.json'),data)
        await worker.release()
        caffeine=(subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
                  if sys.platform=='darwin' else None)
        try:
            schedule=([(mode,item) for item in inputs for mode in modes] if prefix else
                      [(mode,item) for mode in modes for item in inputs])
            for mode,(case_id,original) in schedule:
                if any(r['case_id']==case_id and r['mode']==mode for r in data['results']):
                    continue
                await asyncio.to_thread(guard.wait,False)
                request=deepcopy(original)
                if prefix:
                    request['reuse_prefix']=mode=='prefix'
                await asyncio.wait_for(priority.wait_for_foreground(request),
                                       max(.1,guard.deadline-time.monotonic()-60))
                started=time.monotonic()
                first_variant_seconds=None
                async def progress(texts):
                    nonlocal first_variant_seconds
                    if texts and first_variant_seconds is None:
                        first_variant_seconds=time.monotonic()-started
                if mode=='cold':
                    process=await asyncio.create_subprocess_exec(str(PYTHON),'-m','scripts.local_reply_mlx',cwd=PROJECT,
                        stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
                        env=dict(os.environ,HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1'))
                    try:
                        stdout,stderr=await asyncio.wait_for(process.communicate(json.dumps(request).encode()),min(180,guard.deadline-time.monotonic()-30))
                        if process.returncode and local_reply.evaluation_preempted(request):
                            data['status']='paused_for_reply'
                            return
                        if process.returncode:
                            data['failure_type'] = ('gpu_busy' if b'gpu_busy' in stderr else
                                                    'time_limit' if b'TimeoutError' in stderr else 'worker_exit')
                            raise ValueError('Холодная генерация не выполнена.')
                        receipt=json.loads(stdout)
                    finally:
                        if process.returncode is None:
                            process.terminate()
                            await process.wait()
                else:
                    try:
                        await local_reply._generate(request,on_variant=progress)
                    except ValueError:
                        if request.get('preempted'):
                            data['status']='paused_for_reply'
                            return
                        if not request.get('receipt'):
                            raise
                    receipt=request['receipt']
                    receipt['variants']=[a['text'] for a in receipt['attempts'] if a.get('rejection') is None]
                data['results'].append(dict(mode=mode,case_id=case_id,seconds=time.monotonic()-started,
                    first_variant_seconds=first_variant_seconds,
                    variants_sha256=hashlib.sha256(json.dumps(receipt.get('variants',[]),ensure_ascii=False).encode()).hexdigest(),
                    timings=receipt.get('timings',{}),attempts=receipt['attempts']))
                job['completed_modes']=len(data['results'])
                write_json(root/(stem+'.json'),data)
                write_json(job_path,job)
                print(f"{len(data['results'])}/10: {mode}, {data['results'][-1]['seconds']:.1f} seconds",flush=True)
            for mode in modes:
                data[mode+'_median_seconds']=median(r['seconds'] for r in data['results'] if r['mode']==mode)
            data['first_variant_median_seconds']=median(r['first_variant_seconds'] for r in data['results']
                                                       if r['mode']==modes[1] and r.get('first_variant_seconds') is not None)
            data['identical_outputs']=all(
                next(r['variants_sha256'] for r in data['results'] if r['case_id']==i and r['mode']==modes[0])==
                next(r['variants_sha256'] for r in data['results'] if r['case_id']==i and r['mode']==modes[1])
                for i,_ in inputs)
            data['status']='complete'
            rows=['# Скорость локальной генерации','',
                  'Пять одинаковых запросов, два варианта ответа, одинаковые веса и seed. Без API и обучения.',
                  '', '| Ситуация | Новый процесс, с | Повторное использование, с | Первый локальный вариант, с |',
                  '|---|---:|---:|---:|']
            if prefix:
                rows[4]='| Ситуация | Без кеша префикса, с | С кешем префикса, с | Первый локальный вариант, с |'
            for index,(case_id,_) in enumerate(inputs,1):
                cold=next(r for r in data['results'] if r['case_id']==case_id and r['mode']==modes[0])
                resident=next(r for r in data['results'] if r['case_id']==case_id and r['mode']==modes[1])
                first=resident.get('first_variant_seconds')
                rows.append(f"| {index} | {cold['seconds']:.1f} | {resident['seconds']:.1f} | "
                            +(f'{first:.1f}' if first is not None else '—')+' |')
            rows += ['',f"Медиана: {data[modes[0]+'_median_seconds']:.1f} → {data[modes[1]+'_median_seconds']:.1f} с.",
                     '', 'Время первого варианта включает техническую проверку; смысловая API-проверка добавляет задержку в боте.',
                     'Первый запрос нового резидентного процесса тоже загружает модель. После двух минут простоя модель освобождается.']
            rows += ['', 'Совпадение текстов на всех ситуациях: '+str(data['identical_outputs'])]
            write_text(root/(stem+'.md'),'\n'.join(rows)+'\n')
        except (BudgetExpired,asyncio.TimeoutError):
            data['status']='paused'
        except BaseException:
            data['status']='failed'
            raise
        finally:
            await worker.release()
            if caffeine:
                caffeine.terminate()
                caffeine.wait()
            write_json(root/(stem+'.json'),data)
            job.update(status=data['status'],finished_at=time.time())
            write_json(job_path,job)

async def main(ws,*,prefix=False):
    deadline=time.monotonic()+900
    while deadline-time.monotonic()>90:
        await run(ws,deadline-time.monotonic(),prefix=prefix)
        result=read_json(ws.root/'reply_assistant'/('prefix_benchmark.json' if prefix else 'worker_benchmark.json'),{})
        if result.get('status')!='paused_for_reply':
            return
        print('Живой запрос: замер сохранён, ожидаю освобождения модели.',flush=True)
        await asyncio.sleep(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account',type=int,required=True)
    parser.add_argument('--dialog',type=int,required=True)
    parser.add_argument('--prefix',action='store_true',help='Compare within-request prefix caching on paired requests.')
    args=parser.parse_args()
    asyncio.run(main(resolve_workspace(args.account,args.dialog),prefix=args.prefix))

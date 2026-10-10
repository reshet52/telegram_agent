"""CPU preparation and resumable before/after reply evaluation, without Telegram."""

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path

from mybot.ai.reply_provider import reply_model
from mybot.config import Config
from mybot.services import reply_evaluation as evaluation
from mybot.services.mixed_replies import MixedReplies
from mybot.services.reply_memory import ReplyMemory
from mybot.services.reply_service import ReplyService
from mybot.services.style_versions import selected
from mybot.storage.atomic import read_json, write_json
from mybot.storage.reply_practice import PracticeStore
from scripts.train_reply_adapter import resolve_workspace


async def smoke(ws):
    from mybot.services.online_learning import request_started, request_finished
    from mybot.services.night_training_control import NightGuard
    report = {'version': selected(ws), 'results': [], 'manual_style_assessed': False}
    inputs = [
        ([{'message_id': -1, 'sender': 'Other', 'text': 'Hey❤️', 'type': 'text'}],
         'Ответь естественно, в моей манере.'),
        ([{'message_id': -3, 'sender': 'Other', 'text': 'Hey❤️', 'type': 'text'},
          {'message_id': -2, 'sender': 'Я', 'text': 'Hey❤️', 'type': 'text'},
          {'message_id': -1, 'sender': 'Other', 'text': 'How did your day go?❤️', 'type': 'text'}],
         'Ответь на вопрос о дне. Можешь придумать согласующиеся бытовые детали.'),
        ([{'message_id': -1, 'sender': 'Other', 'text': 'Can we talk tonight?', 'type': 'text'}],
         'Сегодня не могу. Предложи завтра после семи вечера. Коротко по-английски, без эмодзи.'),
    ]
    output = ws.root / 'reply_assistant/smoke.json'
    if output.exists():
        write_json(output.parent / ('smoke_' + str(time.time_ns()) + '.json'), read_json(output))
    guard = NightGuard(20*60, reserve=60)
    request_started(ws)
    try:
        for context, instruction in inputs:
            await asyncio.to_thread(guard.wait, False)
            service = ReplyService(context, {}, workspace=ws)
            generator = MixedReplies(service, version=report['version'],
                facts_builder=lambda c, instruction, *args: json.dumps({'instruction': instruction}, ensure_ascii=False))
            result = await asyncio.wait_for(generator.start(instruction, lambda record: asyncio.sleep(0)),
                                          timeout=max(1, guard.deadline - time.monotonic() - 30))
            report['results'].append(result)
            write_json(output, report)
            print('Синтетическая ситуация', len(report['results']), ': готово', len(result['variants']), flush=True)
    finally:
        request_finished(ws)
    print('Проверка:', output)


async def compare(ws, *, limit=None, minutes=120):
    from mybot.services.night_training_control import NightGuard, BudgetExpired
    data = evaluation.load(ws)
    if not data:
        raise ValueError('Сначала подготовьте проверку командой prepare.')
    evaluation.validate(ws, data)
    version = selected(ws)
    code_root = Path(__file__).resolve().parents[1] / 'mybot'
    fingerprint = hashlib.sha256(b''.join((code_root / name).read_bytes() for name in (
        'services/mixed_replies.py', 'services/reply_memory.py', 'ai/reply_understanding.py',
        'ai/local_reply.py', 'ai/reply_provider.py', 'ai/local_sampling.py', 'ai/reply_contract.py'))).hexdigest()
    settings = {'version': version, 'api_provider': Config.REPLY_PROVIDER, 'api_model': reply_model(),
                'pipeline_sha256': fingerprint, 'local_seed_mode': 'case-id-v1'}
    if data.get('baseline') and data['baseline'] != settings:
        raise ValueError('Модель, API или код конвейера изменились. Продолжение этого сравнения несопоставимо.')
    data['baseline'] = settings
    ws.root.joinpath('reply_assistant').mkdir(parents=True, exist_ok=True)
    job_path = ws.root / 'reply_assistant/comparison_job.json'
    job = {'account_id': ws.account_id, 'dialog_id': ws.dialog_id, 'pid': os.getpid(),
           'status': 'running', 'phase': 'generating', 'started_at': time.time(),
           'completed_modes': sum(len(c['results']) for c in data['cases']), 'total_modes': len(data['cases'])*2}
    def progress(phase, _):
        job['phase'] = phase
        write_json(job_path, job)
    guard = NightGuard(minutes * 60, reserve=60, progress=progress)
    completed = 0
    # The schedule lock prevents two evaluation workers from spending API/GPU twice.
    with (ws.root / 'reply_assistant/evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(job_path, job)
        write_json(ws.root / 'learning/online_pause.json', {'pause': True})
        wake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())]) if sys.platform == 'darwin' else None
        try:
            for item in data['cases']:
                if limit is not None and completed >= limit:
                    break
                case = PracticeStore(ws).get(item['id'])
                context, _ = PracticeStore(ws).source(case)
                def validator():
                    PracticeStore(ws).source(case)
                service = ReplyService(context, {}, workspace=ws)
                for mode in ('baseline', 'enhanced'):
                    if mode in item['results']:
                        continue
                    await asyncio.to_thread(guard.wait, False)
                    job.update(case_id=item['id'], mode=mode, phase='generating')
                    write_json(job_path, job)
                    generator = MixedReplies(service, validator=validator, case_id=case['id'], version=version,
                        facts_builder=lambda c, instruction, *args: json.dumps({'instruction': instruction}, ensure_ascii=False),
                        enhanced=(mode == 'enhanced'), seed=int(case['id'][:8], 16) % 2**30,
                        background_evaluation=True)
                    current = item.setdefault('in_progress', {}).get(mode)
                    if current:
                        record = read_json(generator.path(current))
                        if record and record.get('sides'):
                            record.pop('cancelled', None)
                            record['stale'] = False
                            interrupted = [s for s, v in record['sides'].items() if v['status'] == 'pending']
                            for side in interrupted:
                                record['sides'][side].update(status='failed', error='Проверка была прервана.')
                            generator.save(record)
                            for side in interrupted:
                                if record['sides'][side].get('request'):
                                    await asyncio.wait_for(generator.retry(record['id'], side, lambda r: asyncio.sleep(0)),
                                        timeout=max(1, guard.deadline - time.monotonic() - 30))
                            record = read_json(generator.path(current))
                            while record.get('sides', {}).get('local', {}).get('preempted'):
                                await asyncio.wait_for(generator.retry(record['id'], 'local', lambda r: asyncio.sleep(0)),
                                    timeout=max(1, guard.deadline - time.monotonic() - 30))
                                record = read_json(generator.path(current))
                            # Do not repeat any completed provider, including a rejected result.
                            item['results'][mode] = record
                            evaluation.save_progress(ws, data)
                            continue
                    async def publish(record):
                        item.setdefault('in_progress', {})[mode] = record['id']
                        evaluation.save_progress(ws, data)
                    record = await asyncio.wait_for(generator.start(item['instruction'], publish),
                        timeout=max(1, guard.deadline - time.monotonic() - 30))
                    while record.get('sides', {}).get('local', {}).get('preempted'):
                        await asyncio.wait_for(generator.retry(record['id'], 'local', publish),
                            timeout=max(1, guard.deadline - time.monotonic() - 30))
                        record = read_json(generator.path(record['id']))
                    item['results'][mode] = record
                    evaluation.save_progress(ws, data)
                    job['completed_modes'] = sum(len(c['results']) for c in data['cases'])
                    job['total_modes'] = len(data['cases']) * 2
                    write_json(job_path, job)
                    evaluation.report(ws)
                    print(f"{completed + 1}/{len(data['cases'])}: {mode}, готово {len(record['variants'])}", flush=True)
                completed += 1
            job['status'] = 'complete' if all(len(c['results']) == 2 for c in data['cases']) else 'paused'
        except (BudgetExpired, asyncio.TimeoutError):
            job['status'] = 'paused'
            print('Предел времени: результаты сохранены, команду можно продолжить.', flush=True)
        except BaseException:
            job['status'] = 'interrupted'
            raise
        finally:
            data['pauses'] = {'count': guard.pause_count, 'seconds': guard.pause_seconds}
            evaluation.save_progress(ws, data)
            evaluation.report(ws)
            job.update(completed_modes=sum(len(c['results']) for c in data['cases']),
                       total_modes=len(data['cases'])*2, finished_at=time.time())
            write_json(job_path, job)
            if wake:
                wake.terminate()
                wake.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('command', choices=('prepare', 'compare', 'report', 'import-ratings', 'check-format', 'smoke'))
    parser.add_argument('--ratings')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--minutes', type=float, default=120)
    parser.add_argument('--background', action='store_true', help='Run comparison independently, with progress on disk.')
    args = parser.parse_args()
    if not 1 <= args.minutes <= 1440 or args.limit is not None and args.limit < 1:
        parser.error('Некорректный предел времени/число ситуаций.')
    ws = resolve_workspace(args.account, args.dialog)
    if args.background:
        if args.command != 'compare':
            parser.error('--background доступен только для compare.')
        job_path = ws.root / 'reply_assistant/comparison_job.json'
        prior = read_json(job_path, {})
        if prior.get('status') == 'running':
            try:
                os.kill(prior['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('Сравнение уже выполняется.')
        command = [sys.executable, '-m', 'scripts.reply_assistant', '--account', str(args.account),
                   '--dialog', str(args.dialog), 'compare', '--minutes', str(args.minutes)]
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        from mybot.services.local_training import PROJECT
        log_path = ws.root / 'reply_assistant/comparison.log'
        with log_path.open('ab') as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        print('PID:', process.pid)
        print('Прогресс:', job_path)
        print('Лог:', log_path)
        return
    if args.command == 'prepare':
        coverage = ReplyMemory(ws).sync()
        write_json(ws.root / 'reply_assistant/memory_coverage.json', coverage)
        print(coverage)
        data = evaluation.prepare(ws)
        print('Независимых ситуаций:', len(data['cases']), '/', data['requested_cases'])
        evaluation.report(ws)
    elif args.command == 'compare':
        asyncio.run(compare(ws, limit=args.limit, minutes=args.minutes))
    elif args.command == 'smoke':
        asyncio.run(smoke(ws))
    elif args.command == 'import-ratings':
        if not args.ratings:
            parser.error('Укажите --ratings с файлом оценок из страницы проверки.')
        print(evaluation.import_ratings(ws, args.ratings))
    elif args.command == 'check-format':
        from transformers import AutoTokenizer
        from mybot.services.local_training import MODEL
        from mybot.ai.local_format import prompt_tokens, training_tokens
        from mybot.services.learning_dataset import example
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        data = evaluation.load(ws)
        checked = 0
        for item in data['cases']:
            context, target = PracticeStore(ws).source(PracticeStore(ws).get(item['id']))
            training_tokens(tokenizer, example(context, target))
            for mode, record in item['results'].items():
                request = record.get('sides', {}).get('local', {}).get('request')
                if request:
                    if len(prompt_tokens(tokenizer, request['messages'])) > 8192:
                        raise ValueError('Обогащённый контекст превышает лимит локальной модели.')
                    checked += 1
        result = {'checked_requests': checked, 'status': 'passed' if checked else 'no_generated_requests',
                  'checked_at': time.time(), 'weights_changed': False}
        write_json(ws.root / 'reply_assistant/format_check.json', result)
        print(result)
    else:
        print(evaluation.report(ws))
    print('Страница проверки:', ws.root / 'reply_assistant/evaluation.html')


if __name__ == '__main__':
    main()

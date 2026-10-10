"""Managed local inference shared by lessons, comparisons and live suggestions."""

import asyncio
import json
import os
import secrets
import time

from mybot.services.local_training import PROJECT, PYTHON, MODEL, available
from mybot.services.style_versions import adapter
from mybot.services.learning_dataset import SYSTEM
from mybot.storage.atomic import read_json
from mybot.ai.local_format import FORMAT_VERSION
from mybot.ai.reply_contract import DRAFT_POLICY

_lock = asyncio.Lock()


def evaluation_preempted(request):
    from pathlib import Path
    return (bool(request.get('background_evaluation')) and bool(request.get('workspace_root'))
            and read_json(Path(request['workspace_root']) / 'reply_assistant/ui_request.json', {}).get('at', 0)
            > request.get('evaluation_ui_at', 0))


def live_messages(context, recent):
    from mybot.services.learning_dataset import example
    turns = [dict(m, text=m.get('text') or '[' + m.get('type', 'сообщение') + ']') for m in recent]
    messages = example(turns, [])['prompt']
    messages[0]['content'] += '\n\n' + DRAFT_POLICY
    if context:
        if len(context) > 12000:
            raise ValueError('Блок сведений слишком длинный для локальной модели.')
        messages[0]['content'] += ('\n\nСведения для текущего ответа (не реплики переписки):\n' + context
                                  + '\nЦель и указание заданы владельцем; память может устареть. '
                                  'Отвечай на последние реплики ниже.')
    return messages


def snapshot(prompt, workspace, version, *, messages=None, count=3):
    path = adapter(workspace, version)
    return {'provider': 'mlx', 'model': str(MODEL), 'version': version,
            'workspace_root': str(workspace.root.resolve()), 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
            'format_version': FORMAT_VERSION,
            'adapter': str(path.resolve()) if path else None,
            'expected_adapter_sha256': read_json(path / 'run.json')['adapter_sha256'] if path else None,
            'seed': secrets.randbelow(2**30),
            'messages': messages or [{'role': 'system', 'content': SYSTEM},
                                     {'role': 'user', 'content': prompt}], 'count': count}


async def generate(request, *, on_variant=None):
    from mybot.services.online_learning import request_started, request_finished, pause_for_reply
    from mybot.storage.workspace import Workspace
    from pathlib import Path
    ws = (Workspace(request['account_id'], request['dialog_id'], '', Path(request['workspace_root']))
          if request.get('workspace_root') else None)
    foreground = not request.get('background_evaluation')
    if foreground:
        request_started(ws)
    try:
        if ws:
            await pause_for_reply(ws)
        if on_variant is None:
            return await _generate(request)
        return await _generate(request, on_variant=on_variant)
    finally:
        if foreground:
            request_finished(ws)


async def _generate(request, *, on_variant=None):
    if not available():
        raise ValueError('Локальная модель или окружение MLX недоступны. API автоматически не используется.')
    if _lock.locked():
        raise ValueError('Локальная модель уже занята. Дождитесь завершения запроса.')
    async with _lock:
        started = time.monotonic()
        from mybot.ai.local_worker import worker
        async def progress(event):
            if (event.get('adapter_sha256') != request.get('expected_adapter_sha256')
                    or event.get('format_version') != FORMAT_VERSION):
                raise ValueError('Промежуточный результат относится к другой версии модели.')
            if on_variant:
                await on_variant(event['variants'])
        try:
            result = await asyncio.wait_for(worker.exchange(request, progress), timeout=180)
            if not isinstance(result, dict):
                raise ValueError('Некорректный результат локальной модели.')
            if result.get('adapter_sha256') != request.get('expected_adapter_sha256'):
                await worker.stop()
                raise ValueError('Загруженный адаптер не соответствует выбранной версии.')
            if result.get('format_version') != FORMAT_VERSION:
                await worker.stop()
                raise ValueError('Генератор использовал другой формат ответа.')
            request['receipt'] = {k: result[k] for k in ('seconds', 'adapter_sha256', 'model', 'format_version', 'attempts')}
            if result.get('decision_scores') is not None:
                request['receipt']['decision_scores'] = result['decision_scores']
            request['receipt']['total_seconds'] = time.monotonic() - started
            request['receipt']['timings'] = result.get('timings', {})
            if result.get('error') == 'invalid_generation':
                partial = result.get('variants')
                accepted = {a.get('text') for a in result['attempts'] if a.get('rejection') is None}
                if (request.get('allow_partial') and isinstance(partial, list)
                        and 0 < len(partial) < request['count']
                        and all(isinstance(v, str) and v.strip() and v in accepted for v in partial)
                        and len(set(partial)) == len(partial)):
                    request['receipt']['partial_generation'] = True
                    return partial
                reasons = {a.get('rejection') for a in result['attempts'] if a.get('rejection')}
                if reasons == {'duplicate'}:
                    number = {1: 'один ответ', 2: 'два разных варианта', 3: 'три разных варианта'}[request['count']]
                    raise ValueError(f'Модель повторяет один и тот же ответ: не удалось получить {number}. '
                                     'Попытки исчерпаны. Попробуйте другой отрывок.')
                raise ValueError('Модель вернула некорректный ответ: служебные теги, повторения или незавершённый текст. '
                                 'Попытки исчерпаны. Ответ не используется; можно повторить локальную часть.')
            variants = result.get('variants')
            if (not isinstance(variants, list) or len(variants) != request['count']
                    or any(not isinstance(x, str) or not x.strip() for x in variants)
                    or len(set(variants)) != len(variants)):
                raise ValueError('Некорректный результат локальной модели.')
            return result['variants']
        except asyncio.TimeoutError as error:
            await worker.stop()
            raise ValueError('Локальная модель не ответила за 3 минуты. Запрос остановлен; повторите после освобождения памяти.') from error
        except ValueError:
            if evaluation_preempted(request):
                request['preempted'] = True
                raise ValueError('Сравнение приостановлено ради запроса владельца.')
            raise


async def decide(request):
    request['task'] = 'decision'
    request['count'] = 1
    variants = await generate(request)
    if variants[0] not in {'REPLY', 'WAIT'}:
        raise ValueError('Локальная модель не смогла определить, нужен ли ответ. Попробуйте позже.')
    return variants[0].lower()

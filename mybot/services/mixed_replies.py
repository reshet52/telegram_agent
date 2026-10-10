"""Two independent reply providers sharing one immutable, dialog-scoped request."""

import asyncio
import json
import time
import os
from datetime import datetime, timezone
from copy import deepcopy
from uuid import uuid4

from mybot.ai import local_reply
from mybot.ai.local_context import build_local_context
from mybot.ai.reply_provider import generate_reply, request_snapshot
from mybot.services.message_order import context_snapshot
from mybot.services.style_versions import selected
from mybot.services.reply_memory import ReplyMemory, retrieve
from mybot.ai import reply_understanding
from mybot.storage.atomic import read_json, write_json


def enabled(workspace):
    if workspace is None:
        return False
    state = read_json(workspace.root / 'learning/mixed_replies.json', {})
    if state and (state.get('account_id'), state.get('dialog_id')) != (workspace.account_id, workspace.dialog_id):
        raise ValueError('Настройки смешанных ответов принадлежат другому диалогу.')
    return bool(state.get('enabled'))


def parse_pair(text):
    return parse_variants(text, 2)


def parse_variants(text, count):
    value = json.loads(text.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    variants = value.get('variants') if isinstance(value, dict) else value
    if (not isinstance(variants, list) or len(variants) != count
            or any(not isinstance(v, str) or not v.strip() for v in variants)
            or len({v.strip() for v in variants}) != count):
        raise ValueError('API не вернул нужное количество разных ответов.')
    return [v.strip() for v in variants]


class MixedReplies:
    def __init__(self, service, *, validator=None, case_id=None, facts_builder=None, enhanced=True, version=None,
                 seed=None, background_evaluation=False):
        self.service = service
        self.validator = validator
        self.case_id = case_id
        self.facts_builder = facts_builder or build_local_context
        self.enhanced = enhanced
        self.version = version
        self.seed = seed
        self.background_evaluation = background_evaluation

    def current(self, record):
        if record.get('cancelled'):
            return False
        found = record.get('preparation', {}).get('retrieval', {})
        if found.get('evidence') or found.get('facts') or found.get('style_examples') or found.get('action_examples'):
            from mybot.storage.reply_practice import PracticeStore, digest, source_message
            from mybot.storage.deletions import load_deleted_message_ids
            history = {m['message_id']: m for m in PracticeStore(self.service.workspace).history()}
            deleted = load_deleted_message_ids(self.service.workspace.deleted_message_ids)
            for message in found.get('evidence', []):
                key = message['message_id']
                if key in deleted or history.get(key) != source_message(message):
                    return False
            for fact in found.get('facts', []):
                key = fact['source_id']
                if key in deleted or key not in history or digest([history[key]]) != fact['source_hash']:
                    return False
            for item in found.get('style_examples', []) + found.get('action_examples', []):
                ids = item['source_ids']
                if (set(ids) & deleted or any(i not in history for i in ids)
                        or digest([history[i] for i in ids]) != item.get('source_digest')):
                    return False
        if self.validator:
            try:
                self.validator()
                return record.get('case_id') == self.case_id
            except ValueError:
                return False
        # New messages do not change the situation the owner requested.
        # Validate the original sources instead of comparing to the moving live tail.
        if record.get('snapshot_version') == 1:
            from mybot.storage.reply_practice import PracticeStore, source_message
            from mybot.storage.deletions import load_deleted_message_ids
            history = {m['message_id']: m for m in PracticeStore(self.service.workspace).history()}
            deleted = load_deleted_message_ids(self.service.workspace.deleted_message_ids)
            recent = {m['message_id']: source_message(m) for m in self.service.recent_messages}
            persisted = set(record.get('persisted_context_ids', [m['message_id'] for m in record['context']]))
            for message in record['context']:
                key = message['message_id']
                if key in deleted or (key in persisted and key not in history):
                    return False
                actual = history.get(key, recent.get(key))
                if actual is not None and actual != source_message(message):
                    return False
            return True
        return record['context'] == context_snapshot(self.service.recent_messages)

    async def start(self, instruction, update):
        service = self.service
        async with service.generation_lock:
            context = context_snapshot(service.recent_messages)
            facts = self.facts_builder(context, instruction, service.memory, service.workspace)
            if self.enhanced:
                try:
                    settings = json.loads(facts)
                except ValueError:
                    settings = {}
                if isinstance(settings, dict):
                    # Untimed legacy profile summaries are not source-grounded facts.
                    settings = {key: settings[key] for key in ('goal', 'behavior') if key in settings}
                    settings['instruction'] = instruction
                    facts = json.dumps(settings, ensure_ascii=False)
            record = {'id': uuid4().hex, 'context': context, 'variants': [], 'sides': {},
                      'created_at': datetime.now(timezone.utc).isoformat(),
                      'case_id': self.case_id, 'context_kind': 'historical' if self.case_id else 'live',
                      'account_id': service.workspace.account_id, 'dialog_id': service.workspace.dialog_id,
                      'context_message_id': max((m['message_id'] for m in context), default=None),
                      'reply_time': context[-1].get('date') if self.case_id and context else datetime.now(timezone.utc).isoformat(),
                      'instruction': instruction, 'facts_block': facts, 'enhanced': self.enhanced}
            # Telegram tail refresh can contain sources not persisted by the listener yet.
            # Freeze those too, then validate once they arrive; never rewrite history here.
            from mybot.storage.reply_practice import PracticeStore, source_message
            history = {m['message_id']: m for m in PracticeStore(service.workspace).history()}
            record['snapshot_version'] = 1
            record['persisted_context_ids'] = [m['message_id'] for m in context if m['message_id'] in history]
            record['preparation'] = {'status': 'pending' if self.enhanced else 'disabled', 'audit': []}
            self.save(record)
            await update(deepcopy(record))
            if self.enhanced:
                await self.prepare(record, update)
                if record['preparation']['status'] == 'needs_clarification' or not self.current(record):
                    return record
            self.requests(record)
            self.save(record)
            await update(deepcopy(record))
            await self.run(record, [s for s in ('api', 'local') if record['sides'][s]['status'] == 'pending'], update)
            return record

    async def prepare(self, record, update):
        stage = record['preparation']
        previous_status = stage.get('status')
        stage.update(status='pending', started_at=time.time())
        try:
            if previous_status != 'pending':
                self.save(record)
                await update(deepcopy(record))
            found = await asyncio.to_thread(retrieve, self.service.workspace, record['context'],
                                           record['instruction'], historical=bool(self.case_id))
            stage['retrieval'] = found
            try:
                settings = json.loads(record['facts_block'])
            except ValueError:
                settings = {'settings': record['facts_block']}
            payload = {'recent': record['context'], 'instruction': record['instruction'],
                       'last_incoming': self.last_incoming(record['context']),
                       'reply_time': record['reply_time'],
                       'action_examples': found.get('action_examples', []),
                       'explicit_goal': settings.get('goal'), 'evidence': found['evidence'],
                       'facts': found['facts'], 'style_examples': found['style_examples'],
                       'owner_clarification': record.get('owner_clarification')}
            stage['payload'] = payload
            result = await reply_understanding.understand(payload, stage['audit'])
            stage['result'] = result
            if result['search_queries']:
                initial = found
                found = await asyncio.to_thread(retrieve, self.service.workspace, record['context'],
                    record['instruction'], historical=bool(self.case_id), extra_queries=result['search_queries'])
                stage['retrieval'] = found
                payload.update(evidence=found['evidence'], facts=found['facts'], style_examples=found['style_examples'],
                               action_examples=found.get('action_examples', []))
                if found != initial:
                    # Re-ground the basis when semantic search found previously missing evidence.
                    result = await reply_understanding.understand(payload, stage['audit'])
                    stage['result'] = result
            # Save facts only when backed by indexed actual messages, never the generated basis.
            await asyncio.to_thread(ReplyMemory(self.service.workspace).remember, result['facts'],
                                    record['context'] + found['evidence'])
            stage['status'] = 'needs_clarification' if result.get('question') and not record['variants'] else 'ready'
            stage.pop('error', None)
        except asyncio.CancelledError:
            record['cancelled'] = True
            stage.update(status='failed', error='Подготовка прервана.')
            raise
        except Exception as error:
            stage.update(status='failed', error='Смысловая подготовка недоступна (' + type(error).__name__ + ').')
        finally:
            stage['seconds'] = time.time() - stage['started_at']
            record['stale'] = not self.current(record)
            self.save(record)
            if not record.get('cancelled'):
                await update(deepcopy(record))

    def requests(self, record, *, only=None):
        facts = record['facts_block']
        stage = record.get('preparation', {})
        additions = {}
        additions['reply_time'] = record.get('reply_time')
        found = stage.get('retrieval', {})
        # Keep the local system block bounded without clipping owner instructions.
        additions['evidence'] = self.bounded(found.get('evidence', []), 2600)
        additions['facts'] = self.bounded(found.get('facts', []), 900)
        additions['style_examples'] = self.bounded(found.get('style_examples', []), 1600)
        additions['action_examples'] = self.bounded(found.get('action_examples', []), 1400)
        if stage.get('status') == 'ready':
            additions['understanding'] = {k: stage['result'].get(k) for k in
                ('reply_to', 'intent', 'basis', 'response_action', 'action_reason', 'mood')}
        if record.get('owner_clarification'):
            additions['owner_clarification'] = record['owner_clarification']
        if record.get('enhanced'):
            prefix = '\n\nДанные помощника: '
            # Drop whole optional records, never cut an instruction or factual quote.
            while len(facts + prefix + json.dumps(additions, ensure_ascii=False)) > 12000:
                optional = next((key for key in ('style_examples', 'action_examples', 'evidence', 'facts') if additions.get(key)), None)
                if optional:
                    additions[optional].pop()
                elif additions.get('understanding', {}).get('intent'):
                    additions['understanding'].pop('intent')
                elif additions.get('understanding', {}).get('reply_to'):
                    additions['understanding'].pop('reply_to')
                elif additions.get('understanding'):
                    additions.pop('understanding')
                else:
                    break
            facts += prefix + json.dumps(additions, ensure_ascii=False)
        messages = local_reply.live_messages(facts, record['context'])
        if record.get('enhanced'):
            messages[0]['content'] += ('\nСмысловая основа помощника — предложение содержания, '
                'а не готовый текст: вырази его в манере владельца. Явное указание владельца '
                'и актуальные сведения с источниками важнее этой основы. Исторические примеры '
                'показывают стиль; их события не нужно переносить в сегодняшний ответ.')
        version = self.version or selected(self.service.workspace)
        for side in only or ('api', 'local'):
            try:
                if side == 'api':
                    request = request_snapshot(json.dumps({'messages': messages}, ensure_ascii=False), count=2)
                else:
                    request = local_reply.snapshot(facts, self.service.workspace, version, messages=messages, count=2)
                    request['diverse'] = True
                    request['allow_partial'] = bool(record.get('enhanced'))
                    if self.seed is not None:
                        request['seed'] = self.seed
                    request['background_evaluation'] = self.background_evaluation
                record['sides'][side] = {'status': 'pending', 'request': request}
            except ValueError as error:
                record['sides'][side] = {'status': 'failed', 'error': str(error), 'request': None}

    @staticmethod
    def bounded(items, maximum):
        result = []
        for item in items:
            if len(json.dumps(result + [item], ensure_ascii=False)) <= maximum:
                result.append(item)
        return result

    @staticmethod
    def last_incoming(context):
        end = next((i + 1 for i in range(len(context) - 1, -1, -1) if context[i].get('sender') != 'Я'), 0)
        start = end
        while start and context[start - 1].get('sender') != 'Я':
            start -= 1
        return context[start:end]

    async def clarify(self, key, text, update):
        async with self.service.generation_lock:
            record = read_json(self.path(key))
            ws = self.service.workspace
            if (not record or (record['account_id'], record['dialog_id']) != (ws.account_id, ws.dialog_id)
                    or record.get('preparation', {}).get('status') != 'needs_clarification' or not self.current(record)):
                raise ValueError('Ситуация изменилась. Запросите новые варианты.')
            if not text.strip() or len(text) > 2000:
                raise ValueError('Уточнение должно содержать от 1 до 2000 символов.')
            record['owner_clarification'] = text.strip()
            await self.prepare(record, update)
            self.requests(record)
            self.save(record)
            await update(deepcopy(record))
            await self.run(record, [s for s in record['sides'] if record['sides'][s]['status'] == 'pending'], update)
            return record

    def path(self, key):
        if len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Некорректная генерация.')
        return self.service.workspace.root / 'learning/mixed_requests' / (key + '.json')

    def save(self, record):
        from mybot.storage.reply_journal import save_mixed_generation
        write_json(self.path(record['id']), record)
        save_mixed_generation(self.service.workspace, record)

    async def retry(self, key, side, update):
        async with self.service.generation_lock:
            record = read_json(self.path(key))
            ws = self.service.workspace
            if (not record or (record['account_id'], record['dialog_id']) != (ws.account_id, ws.dialog_id)
                    or side not in ('api', 'local', 'understanding')
                    or (record.get('preparation', {}).get('status') != 'failed' if side == 'understanding'
                        else record['sides'].get(side, {}).get('status') != 'failed')):
                raise ValueError('Повтор этой части недоступен.')
            if not self.current(record):
                raise ValueError('Переписка изменилась. Запросите новые варианты.')
            if side == 'understanding':
                await self.prepare(record, update)
                # Successful provider results and their exact requests remain intact.
                failed = [s for s, value in record['sides'].items() if value['status'] == 'failed'
                          and not any(v['source'] == s for v in record['variants'])]
                if failed and record['preparation']['status'] != 'needs_clarification':
                    self.requests(record, only=failed)
                    await self.run(record, [s for s in failed if record['sides'][s]['status'] == 'pending'], update)
                return
            if record['sides'][side]['request'] is None:
                raise ValueError('Локальный адаптер недоступен. Проверьте версию модели.')
            state = record['sides'][side]
            state.setdefault('retries', []).append(deepcopy({k: v for k, v in state.items() if k != 'retries'}))
            state.pop('preempted', None)
            state['request'].pop('preempted', None)
            count = sum(not any(v['number'] == n for v in record['variants'])
                        for n in (range(3, 5) if side == 'api' else range(1, 3)))
            if side == 'api':
                state['request'] = request_snapshot(state['request']['input'], count=count)
            else:
                state['request']['count'] = count
                prior = [v['text'] for v in record['variants'] if v['source'] == side]
                if prior:
                    state['request']['messages'][0]['content'] += '\nУже принят другой вариант, не повторяй его: ' + json.dumps(prior, ensure_ascii=False)
            if side == 'local':
                import secrets
                state['request']['seed'] = self.seed if self.seed is not None else secrets.randbelow(2**30)
                state['request'].pop('receipt', None)
            record['sides'][side]['status'] = 'pending'
            self.save(record)
            await update(deepcopy(record))
            await self.run(record, [side], update)

    async def run(self, record, sides, update):
        from mybot.ai.client import ai_client
        publish_lock = asyncio.Lock()

        async def part(side):
            state = record['sides'][side]
            started = time.monotonic()
            state['started_at'] = datetime.now(timezone.utc).isoformat()
            numbers = [n for n in (range(3, 5) if side == 'api' else range(1, 3))
                       if not any(v['number'] == n for v in record['variants'])]
            async def progress(texts):
                pending = [{'number': n, 'text': t} for n, t in zip(numbers, texts)
                           if not any(v['number'] == n for v in record['variants'])
                           and not any(v['source'] == 'local' and v['text'] == t for v in record['variants'])
                           and not any(c['number'] == n and c['text'] == t for c in state.get('progress_checks', []))]
                if not pending:
                    return
                if record.get('enhanced'):
                    payload = dict(record['preparation']['payload'], owner_clarification=record.get('owner_clarification'))
                    measured = time.monotonic()
                    checked = await reply_understanding.check(payload, pending, state.setdefault('check_audit', []))
                    state['semantic_check_seconds'] = state.get('semantic_check_seconds', 0) + time.monotonic() - measured
                    state.setdefault('checks', []).append(checked)
                    checks = {c['number']: c for c in checked['checks']}
                else:
                    checks = {p['number']: {'accepted': True, 'reason': ''} for p in pending}
                async with publish_lock:
                    for item in pending:
                        state.setdefault('progress_checks', []).append(dict(item, check=checks[item['number']]))
                        if checks[item['number']]['accepted']:
                            record['variants'].append(dict(item, source='local', model=state['request']['model'],
                                version=state['request'].get('version'), seconds=time.monotonic() - started))
                    record['variants'].sort(key=lambda v: v['number'])
                    record['stale'] = not self.current(record)
                    self.save(record)
                    if not record.get('cancelled'):
                        await update(deepcopy(record))
            try:
                if side == 'api':
                    count = state['request'].get('count', 2)
                    result = parse_variants(await generate_reply(state['request']['input'], ai_client, count=count), count)
                else:
                    from mybot.services.online_learning import pause_for_reply
                    if self.background_evaluation:
                        await self.wait_for_foreground(state['request'])
                    await pause_for_reply(self.service.workspace)
                    result = await local_reply.generate(state['request'], on_variant=progress)
                # A partial retry must not replace a successful numbered variant.
                candidates = [{'number': number, 'text': text} for number, text in zip(numbers, result)
                              if not any(v['number'] == number for v in record['variants'])]
                remaining_numbers = [n for n in numbers if not any(v['number'] == n for v in record['variants'])]
                accepted_texts = {v['text'] for v in record['variants'] if v['source'] == side}
                candidates = [v for v in candidates if v['text'] not in accepted_texts]
                if record.get('enhanced'):
                    candidates = await self.checked(record, state, side, candidates)
                unique = []
                for item in candidates:
                    if item['text'] not in accepted_texts:
                        unique.append(item)
                        accepted_texts.add(item['text'])
                candidates = unique
                state.update(status='ready', seconds=time.monotonic() - started)
                state.pop('error', None)
                for item in candidates:
                    record['variants'].append({'number': item['number'], 'text': item['text'], 'source': side,
                                              'model': state['request']['model'],
                                              'version': state['request'].get('version'),
                                              'seconds': state['seconds']})
                if len(candidates) != len(remaining_numbers):
                    state.update(status='failed', error='Некоторые варианты не прошли проверку смысла и недоступны.')
            except asyncio.CancelledError:
                record['cancelled'] = True
                state.update(status='failed', error='Генерация прервана.')
                raise
            except Exception as error:
                state.update(status='failed', error=str(error), seconds=time.monotonic() - started)
                if state.get('request', {}).get('preempted'):
                    state['preempted'] = True
            finally:
                async with publish_lock:
                    state['finished_at'] = datetime.now(timezone.utc).isoformat()
                    record['stale'] = not self.current(record)
                    record['variants'].sort(key=lambda v: v['number'])
                    self.save(record)
                    if not record.get('cancelled'):
                        await update(deepcopy(record))
        await asyncio.gather(*(part(side) for side in sides))

    async def wait_for_foreground(self, request):
        ws = self.service.workspace
        while True:
            pulse = read_json(ws.root / 'reply_assistant/ui_request.json', {})
            active = read_json(ws.root / 'learning/online_request.json', {})
            busy = active.get('active') and active.get('pid') != os.getpid() and time.time() - active.get('at', 0) < 600
            if not busy and time.time() - pulse.get('at', 0) >= 15:
                request['evaluation_ui_at'] = pulse.get('at', 0)
                job = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
                if job.get('pid') == os.getpid():
                    job['phase'] = 'generating'
                    write_json(ws.root / 'reply_assistant/comparison_job.json', job)
                return
            job = read_json(ws.root / 'reply_assistant/comparison_job.json', {})
            if job.get('pid') == os.getpid():
                job['phase'] = 'waiting_for_reply'
                write_json(ws.root / 'reply_assistant/comparison_job.json', job)
            await asyncio.sleep(1)

    async def checked(self, record, state, side, candidates):
        payload = record['preparation'].get('payload', {'recent': record['context'], 'instruction': record['instruction'],
                                                     'last_incoming': self.last_incoming(record['context'])})
        payload = dict(payload, owner_clarification=record.get('owner_clarification'))
        audit = state.setdefault('check_audit', [])
        cached = {c['number']: c['check'] for c in state.get('progress_checks', [])
                  if any(v['number'] == c['number'] and v['text'] == c['text'] for v in candidates)}
        measured = time.monotonic()
        result = await reply_understanding.check(payload, [c for c in candidates if c['number'] not in cached], audit)
        result['checks'] += list(cached.values())
        state['semantic_check_seconds'] = state.get('semantic_check_seconds', 0) + time.monotonic() - measured
        state.setdefault('checks', []).append(result)
        checks = {v['number']: v for v in result['checks']}
        accepted = [v for v in candidates if checks[v['number']]['accepted']]
        rejected = [v for v in candidates if not checks[v['number']]['accepted']]
        if not rejected:
            return accepted
        request = deepcopy(state['request'])
        reasons = [{'number': v['number'], 'text': v['text'], 'reason': checks[v['number']]['reason']} for v in rejected]
        fix = '\n\nПерепиши только неудачные ответы. Сначала ответь на последнюю реплику. Ошибки: ' + json.dumps(reasons, ensure_ascii=False)
        if side == 'local':
            request['count'] = len(rejected)
            request['messages'][0]['content'] += fix
        else:
            request = request_snapshot(request['input'] + fix, count=len(rejected))
        state.setdefault('rewrite_requests', []).append(request)
        try:
            if side == 'local':
                if self.background_evaluation:
                    await self.wait_for_foreground(request)
                texts = await local_reply.generate(request)
            else:
                from mybot.ai.client import ai_client
                value = reply_understanding.parse_json(await generate_reply(request['input'], ai_client, count=len(rejected)))
                texts = value.get('variants', [])
        except Exception as error:
            state.setdefault('rewrite_errors', []).append(type(error).__name__)
            if request.get('preempted'):
                state['preempted'] = True
                state['request']['preempted'] = True
            return accepted
        if len(texts) != len(rejected) or any(not isinstance(t, str) for t in texts):
            return accepted
        revised = [{'number': v['number'], 'text': text} for v, text in zip(rejected, texts)]
        result = await reply_understanding.check(payload, revised, audit)
        state['checks'].append(result)
        checks = {v['number']: v for v in result['checks']}
        return accepted + [v for v in revised if checks[v['number']]['accepted']]

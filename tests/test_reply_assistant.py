"""Offline source, intent, rewriting, clarification and evaluation regressions."""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.ai.reply_understanding import UNDERSTAND, CHECK, understand, check
from mybot.services.mixed_replies import MixedReplies
from mybot.services.reply_memory import ReplyMemory, retrieve
from mybot.services import reply_evaluation as evaluation
from mybot.services.online_dataset import reserved_ids
from mybot.storage.atomic import read_json, write_json
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.workspace import Workspace


class AssistantTests(unittest.IsolatedAsyncioTestCase):
    def test_evidence_index_skips_unchanged_history_but_reconciles_edits_and_deletions(self):
        memory = ReplyMemory(self.ws)
        with patch.object(memory, 'sync', wraps=memory.sync) as sync:
            memory.sync_if_changed()
            memory.sync_if_changed()
            self.assertEqual(sync.call_count, 1)
            self.messages[0]['text'] = 'an edited fact'
            self.save()
            memory.sync_if_changed()
            self.assertEqual(sync.call_count, 2)
            write_json(self.ws.deleted_message_ids, [1])
            memory.sync_if_changed()
            self.assertEqual(sync.call_count, 3)
        self.assertEqual(memory.search('edited', 120), [])

    def test_wait_feedback_is_scoped_and_not_visible_before_confirmation(self):
        store = PracticeStore(self.ws)
        context, response = self.messages[-2:-1], self.messages[-1:]
        case = store.create_from(context, response, 'simple_lesson')
        store.confirm_action(case['id'], 'wait')
        found = retrieve(self.ws, self.messages[-2:], 'day')
        self.assertEqual(len(found['action_examples']), 1)
        self.assertEqual(found['action_examples'][0]['action'], 'wait')
        old = retrieve(self.ws, self.messages[-2:], 'day', historical=True)
        self.assertEqual(old['action_examples'], [])
        record = {'id': 'synthetic', 'context': self.messages[-2:], 'preparation': {'retrieval': found}}
        self.assertTrue(MixedReplies(self.service, validator=lambda: None).current(record))
        self.messages[-2]['text'] = 'changed source'
        self.save()
        self.assertFalse(MixedReplies(self.service, validator=lambda: None).current(record))

    async def test_background_generation_does_not_overwrite_foreground_activity(self):
        from mybot.ai import local_reply
        request = {'workspace_root': str(self.ws.root), 'background_evaluation': True, 'account_id': 1, 'dialog_id': 2}
        with patch('mybot.ai.local_reply._generate', AsyncMock(return_value=['ok'])), \
             patch('mybot.services.online_learning.request_started') as started, \
             patch('mybot.services.online_learning.request_finished') as finished:
            self.assertEqual(await local_reply.generate(request), ['ok'])
            started.assert_not_called()
            finished.assert_not_called()

    async def test_background_worker_stops_only_for_new_owner_request(self):
        from mybot.ai.local_reply import evaluation_preempted
        write_json(self.ws.root / 'reply_assistant/ui_request.json', {'at': 10})
        request = {'workspace_root': str(self.ws.root), 'background_evaluation': True, 'evaluation_ui_at': 10}
        self.assertFalse(evaluation_preempted(request))
        write_json(self.ws.root / 'reply_assistant/ui_request.json', {'at': 11})
        self.assertTrue(evaluation_preempted(request))
        request['background_evaluation'] = False
        self.assertFalse(evaluation_preempted(request))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'synthetic', Path(self.temp.name))
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.messages = [dict(message_id=i + 1, sender='Other' if i % 2 == 0 else 'Я',
            text='How did your day go?' if i % 2 == 0 else f'OWNER_RESPONSE_{i}', type='text',
            date=(start + timedelta(hours=i//2*5, seconds=i % 2)).isoformat()) for i in range(120)]
        self.save()
        self.service = NS(workspace=self.ws, recent_messages=[self.messages[-2]], memory={}, generation_lock=asyncio.Lock())
        self.patches = [patch('mybot.services.mixed_replies.selected', return_value='base'),
            patch('mybot.services.mixed_replies.local_reply.snapshot',
                  side_effect=lambda *a, **k: dict(model='local', version='base', **k)),
            patch('mybot.services.mixed_replies.build_local_context', return_value='{"goal":"explicit goal"}'),
            patch('mybot.services.online_learning.pause_for_reply', AsyncMock())]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.question = None
        self.foundation_inputs = []

    def save(self):
        self.ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in self.messages))

    async def api(self, request, client):
        payload = json.loads(request['input'])
        if request['instructions'] == UNDERSTAND:
            self.foundation_inputs.append(payload)
            return json.dumps({'reply_to': 'Answer how the day went', 'intent': 'Share warmly',
                'basis': 'Describe a quiet day, then ask about theirs', 'missing': [],
                'question': self.question if not payload.get('owner_clarification') else None,
                'search_queries': ['day quiet'], 'facts': []})
        self.assertEqual(request['instructions'], CHECK)
        return json.dumps({'checks': [dict(number=v['number'], answers_last=v['text'] != 'Hey',
            follows_instruction=True, consistent=True, reason='Answer the question' if v['text'] == 'Hey' else '')
            for v in payload['variants']]})

    def test_index_all_sources_resume_edits_deletions_and_identity(self):
        self.messages[0]['type'] = 'photo'
        self.messages[1]['text'] = ''
        self.save()
        memory = ReplyMemory(self.ws)
        def stopped(done, total):
            raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError):
            memory.sync(batch_size=20, progress=stopped)
        result = memory.sync()
        self.assertEqual(result['total'], 120)
        self.assertEqual(result['counts']['media_without_verified_text'], 1)
        self.assertEqual(result['counts']['empty_text'], 1)
        self.assertEqual(memory.sync(), result)
        self.messages[2]['text'] = 'I live in Paris'
        self.save()
        memory.sync()
        self.assertEqual(memory.search('Paris', 120)[0]['message_id'], 3)
        self.messages[2]['text'] = 'I moved to Berlin'
        self.save()
        memory.sync()
        self.assertEqual(memory.search('Paris', 120), [])
        write_json(self.ws.deleted_message_ids, [3])
        memory.sync()
        self.assertEqual(memory.search('Berlin', 120), [])
        with memory.connection() as db:
            self.assertGreater(db.execute('SELECT COUNT(*) FROM revisions WHERE message_id=3').fetchone()[0], 0)
        with self.assertRaises(ValueError):
            ReplyMemory(Workspace(99, 2, 'foreign', self.ws.root)).sync()
        self.messages.pop()
        self.save()
        with self.assertRaisesRegex(ValueError, 'восстановление'):
            memory.sync()

    def test_sent_facts_are_versioned_and_drafts_never_become_facts(self):
        self.messages[1]['text'] = 'I live in Paris'
        self.messages[3]['text'] = 'I moved to Berlin'
        self.save()
        memory = ReplyMemory(self.ws)
        memory.sync()
        memory.remember([dict(subject='owner', property='residence', quote='I live in Paris', source_id=2),
            dict(subject='owner', property='residence', quote='I moved to Berlin', source_id=4),
            dict(subject='owner', property='fiction', quote='I won a lottery', source_id=4)], self.messages)
        self.assertEqual(memory.facts('residence', 2)[0]['quote'], 'I live in Paris')
        self.assertEqual(memory.facts('residence', 4)[0]['quote'], 'I moved to Berlin')
        self.assertEqual(memory.facts('Paris', 4)[0]['quote'], 'I moved to Berlin')
        self.assertEqual(memory.facts('Paris', 4)[0]['source_date'], self.messages[3]['date'])
        self.assertEqual(memory.facts('lottery', 120), [])
        memory.remember([dict(subject='owner', property='draft', quote='UNSENT DRAFT', source_id=999)],
                        [dict(message_id=999, sender='Я', text='UNSENT DRAFT')])
        self.assertEqual(memory.facts('draft', 999), [])
        write_json(self.ws.deleted_message_ids, [4])
        memory.sync()
        self.assertEqual(memory.facts('residence', 4)[0]['quote'], 'I live in Paris')

    async def test_grounding_shared_requests_and_hidden_response(self):
        hidden = self.messages[-1]['text']
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["Quiet day, yours?","Pretty good today"]}')), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['Quiet day ❤️', 'Pretty good ❤️'])):
            record = await MixedReplies(self.service, case_id='a'*32, validator=lambda: None).start('no promises', AsyncMock())
        self.assertEqual([v['number'] for v in record['variants']], [1, 2, 3, 4])
        self.assertEqual(record['preparation']['status'], 'ready')
        self.assertEqual(self.foundation_inputs[0]['explicit_goal'], 'explicit goal')
        requests = [record['preparation']['audit'], record['sides']]
        self.assertNotIn(hidden, json.dumps(requests))
        local = record['sides']['local']['request']['messages']
        self.assertEqual(local, json.loads(record['sides']['api']['request']['input'])['messages'])
        self.assertIn('Describe a quiet day', local[0]['content'])

    async def test_one_rewrite_and_rejected_number_unavailable(self):
        local = AsyncMock(side_effect=[['Hey', 'A quiet day'], ['Hey']])
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["Fine today","A nice day"]}')), \
             patch('mybot.services.mixed_replies.local_reply.generate', local):
            record = await MixedReplies(self.service).start('instruction', AsyncMock())
        self.assertEqual([v['number'] for v in record['variants']], [2, 3, 4])
        self.assertEqual(record['sides']['local']['status'], 'failed')
        self.assertEqual(local.await_count, 2)
        self.assertEqual(local.await_args_list[1].args[0]['count'], 1)
        self.assertEqual(len(record['sides']['local']['rewrite_requests']), 1)
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['A better day'])) as retry, \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock()) as api:
            await MixedReplies(self.service).retry(record['id'], 'local', AsyncMock())
        retry.assert_awaited_once()
        self.assertEqual(retry.await_args.args[0]['count'], 1)
        api.assert_not_awaited()

    async def test_incremental_local_checks_are_cached_and_rewrite_only_rejected_number(self):
        published = []
        async def publish(record):
            published.append([v['number'] for v in record['variants']])
        async def local(request, on_variant=None):
            if on_variant:
                await on_variant(['A quiet day'])
                self.assertTrue(any(1 in ids and 2 not in ids for ids in published))
                await on_variant(['A quiet day', 'Hey'])
                return ['A quiet day', 'Hey']
            self.assertEqual(request['count'], 1)
            return ['A better day']
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["Fine today","A nice day"]}')), \
             patch('mybot.services.mixed_replies.local_reply.generate', side_effect=local) as generate:
            record = await MixedReplies(self.service).start('instruction', publish)
        self.assertEqual([v['number'] for v in record['variants']], [1, 2, 3, 4])
        self.assertEqual(generate.await_count, 2)
        self.assertEqual(len(record['sides']['local']['rewrite_requests']), 1)
        self.assertEqual(record['sides']['local']['status'], 'ready')

    async def test_home_cancels_clarification_but_new_message_keeps_snapshot(self):
        from mybot.telegram.bot_mixed_replies import cancel_clarification
        self.question = 'Are you free tomorrow?'
        mixed = MixedReplies(self.service)
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api):
            record = await mixed.start('instruction', AsyncMock())
        bot = NS(workspace=self.ws, reply_service=self.service,
                 reply_clarification={'key': record['id'], 'workspace': str(self.ws.root)})
        cancel_clarification(bot)
        self.assertIsNone(bot.reply_clarification)
        with self.assertRaises(ValueError):
            await mixed.clarify(record['id'], 'yes', AsyncMock())
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api):
            record = await mixed.start('instruction', AsyncMock())
        self.service.recent_messages.append(dict(message_id=121, sender='Other', text='new'))
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['one', 'two'])), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["three","four"]}')):
            result = await mixed.clarify(record['id'], 'yes', AsyncMock())
        self.assertEqual(result['context'], record['context'])
        self.assertFalse(result['stale'])

    async def test_deleted_retrieved_source_invalidates_selection(self):
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["Nice day","A good day"]}')), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['Quiet day', 'Fine day'])):
            mixed = MixedReplies(self.service)
            record = await mixed.start('instruction', AsyncMock())
        evidence = record['preparation']['retrieval']['evidence']
        self.assertTrue(evidence)
        write_json(self.ws.deleted_message_ids, [evidence[0]['message_id']])
        self.assertFalse(mixed.current(record))

    async def test_clarify_once_only_current_situation(self):
        self.question = 'Can you meet at seven tomorrow?'
        mixed = MixedReplies(self.service)
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["Tomorrow at seven","Seven works"]}')) as api, \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['Seven ❤️', 'Tomorrow ❤️'])) as local:
            record = await mixed.start('instruction', AsyncMock())
            self.assertEqual(record['preparation']['status'], 'needs_clarification')
            api.assert_not_awaited()
            local.assert_not_awaited()
            record = await mixed.clarify(record['id'], 'Yes, seven works', AsyncMock())
            self.assertEqual(len(record['variants']), 4)
            self.assertEqual(record['owner_clarification'], 'Yes, seven works')
        self.assertFalse((self.ws.root / 'dialog_goal.json').exists())

    async def test_api_understanding_failure_preserves_local_and_successful_api_not_repeated(self):
        api = AsyncMock(return_value='{"variants":["Quiet day","A good day"]}')
        with patch('mybot.ai.reply_understanding.execute_request', AsyncMock(side_effect=OSError('offline'))), \
             patch('mybot.services.mixed_replies.generate_reply', api), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['Nice day', 'Fine day'])):
            mixed = MixedReplies(self.service)
            record = await mixed.start('instruction', AsyncMock())
        self.assertEqual(len(record['variants']), 4)
        self.assertEqual(record['preparation']['status'], 'failed')
        self.assertEqual(record['sides']['local']['checks'][0]['status'], 'unavailable')
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=self.api), \
             patch('mybot.services.mixed_replies.generate_reply', api):
            await mixed.retry(record['id'], 'understanding', AsyncMock())
        api.assert_awaited_once()

    async def test_source_changed_during_preparation_skips_generation(self):
        async def changed(request, client):
            result = await self.api(request, client)
            self.messages[-2]['text'] = 'edited original source'
            self.save()
            return result
        with patch('mybot.ai.reply_understanding.execute_request', side_effect=changed), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock()) as local:
            record = await MixedReplies(self.service).start('instruction', AsyncMock())
        self.assertTrue(record['stale'])
        local.assert_not_awaited()

    def test_evaluation_reserves_fifty_sources_and_owner_ratings_are_required(self):
        data = evaluation.prepare(self.ws)
        self.assertEqual(len(data['cases']), 50)
        self.assertEqual(evaluation.prepare(self.ws)['id'], data['id'])
        self.assertEqual(reserved_ids(self.ws), evaluation.reserved(self.ws))
        self.assertEqual(evaluation.report(self.ws)['target_met'], False)
        text = (self.ws.root / 'reply_assistant/evaluation.html').read_text()
        self.assertNotIn('OWNER_RESPONSE_', text)
        self.assertIn(".join('\\n\\n')", text)
        self.assertIn("v.source+'\\n'+v.text", text)
        self.assertNotIn(".join('\n\n')", text)
        source = data['cases'][0]['source_ids'][0]
        write_json(self.ws.root / 'learning/reviewed' / ('a'*32+'.json'), {'examples': {'train': [{'ids': [source]}]}})
        write_json(self.ws.root / 'learning/mlx_runs/r/run.json',
                   dict(account_id=1, dialog_id=2, completed_steps=1, dataset_id='a'*32))
        with self.assertRaises(ValueError):
            evaluation.validate(self.ws, data)

    def test_confirmed_correction_retrieved_but_not_from_future(self):
        store = PracticeStore(self.ws)
        case = store.create_from([self.messages[0]], [self.messages[1]], 'simple_lesson')
        evidence = dict(account_id=1, dialog_id=2, case_id=case['id'], response_ids=[2], context_ids=[1],
                        text='CORRECTED STYLE', confirmed_at='2026-01-01T00:00:00+00:00')
        write_json(self.ws.root / 'learning/confirmed/a.json', evidence)
        current = retrieve(self.ws, [self.messages[-2]], 'day')
        self.assertTrue(any(v['reply'] == 'CORRECTED STYLE' for v in current['style_examples']))
        historical = retrieve(self.ws, [self.messages[-2]], 'day', historical=True)
        self.assertFalse(any(v['reply'] == 'CORRECTED STYLE' for v in historical['style_examples']))

    async def test_checker_failure_is_visible_and_bad_schema_not_trusted(self):
        with patch('mybot.ai.reply_understanding.execute_request', AsyncMock(return_value='{"checks":[]}')):
            result = await check({}, [{'number': 1, 'text': 'Okay'}], [])
        self.assertEqual(result['status'], 'unavailable')
        with patch('mybot.ai.reply_understanding.execute_request', AsyncMock(return_value='{}')):
            with self.assertRaises(ValueError):
                await understand({}, [])

    def test_ratings_survive_worker_updates_and_target_requires_owner_assessment(self):
        data = evaluation.prepare(self.ws)
        for item in data['cases']:
            item['results'] = {mode: {'variants': [{'number': 1, 'source': 'api', 'text': 'reply'}]}
                               for mode in ('baseline', 'enhanced')}
        evaluation.save_progress(self.ws, data)
        self.assertFalse(evaluation.report(self.ws)['target_met'])
        ratings = {c['id']: {mode: {'1': {'rating': 'ready', 'meaning_error': False, 'contradiction': False}}
                            for mode in ('baseline', 'enhanced')} for c in data['cases']}
        output = self.ws.root / 'owner-ratings.json'
        write_json(output, {'evaluation_id': data['id'], 'ratings': ratings})
        self.assertTrue(evaluation.import_ratings(self.ws, output)['target_met'])
        evaluation.save_progress(self.ws, data)  # Simulate a worker's older in-memory snapshot.
        self.assertTrue(evaluation.report(self.ws)['target_met'])
        ratings[data['cases'][0]['id']]['enhanced']['1']['contradiction'] = True
        write_json(output, {'evaluation_id': data['id'], 'ratings': ratings})
        self.assertFalse(evaluation.import_ratings(self.ws, output)['target_met'])

    def test_last_incoming_target_is_not_owners_latest_message(self):
        context = [{'message_id': 1, 'sender': 'Other', 'text': 'How was your day?'},
                   {'message_id': 2, 'sender': 'Other', 'text': 'Are you tired?'},
                   {'message_id': 3, 'sender': 'Я', 'text': 'Earlier reply'}]
        self.assertEqual(MixedReplies.last_incoming(context), context[:2])

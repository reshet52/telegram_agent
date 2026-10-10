import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mybot.services.mixed_replies import MixedReplies, parse_pair
from mybot.storage.reply_journal import create_draft, get_draft, get_generation, update_draft_text
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_mixed_replies import card


class MixedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'test', Path(self.temp.name))
        self.messages = [dict(message_id=1, sender='Other', text='question', type='text')]
        self.service = SimpleNamespace(workspace=self.ws, recent_messages=self.messages,
                                       memory={}, generation_lock=asyncio.Lock())
        # These concurrency/draft regressions isolate provider execution.
        self.mixed = MixedReplies(self.service, enhanced=False)
        self.patches = [patch('mybot.services.mixed_replies.build_local_context', return_value='facts'),
                        patch('mybot.services.mixed_replies.selected', return_value='base'),
                        patch('mybot.services.mixed_replies.local_reply.snapshot',
                              side_effect=lambda *a, **kw: dict(provider='mlx', model='local', version='base', **kw)),
                        patch('mybot.services.online_learning.pause_for_reply', new_callable=AsyncMock)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    async def test_api_visible_while_local_waits_and_fourth_editable(self):
        gate = asyncio.Event()
        published = []
        async def local(request, **kwargs):
            await gate.wait()
            return ['local1', 'local2']
        async def publish(value):
            published.append(value)
            if value['sides'].get('api', {}).get('status') == 'ready' and value['sides']['local']['status'] == 'pending':
                self.assertEqual([v['number'] for v in value['variants']], [3, 4])
                gate.set()
        with patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["api1","api2"]}')) as api, patch('mybot.services.mixed_replies.local_reply.generate', local):
            result = await self.mixed.start('instruction', publish)
        self.assertEqual([v['number'] for v in result['variants']], [1, 2, 3, 4])
        api.assert_awaited_once()
        draft = create_draft(self.ws, result['id'], 4)
        update_draft_text(self.ws, draft, 'edited API')
        self.assertEqual(get_draft(self.ws, draft)['final_text'], 'edited API')
        self.assertEqual(get_generation(self.ws, result['id'])['variants'][3]['source'], 'api')
        api_messages = json.loads(result['sides']['api']['request']['input'])['messages']
        self.assertEqual(api_messages, result['sides']['local']['request']['messages'])
        from mybot.ai.reply_contract import DRAFT_POLICY
        self.assertIn(DRAFT_POLICY, api_messages[0]['content'])
        self.assertIn(DRAFT_POLICY, result['sides']['api']['request']['instructions'])

    async def test_retry_local_does_not_repeat_api(self):
        with patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["api1","api2"]}')) as api, patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(side_effect=ValueError('failed'))):
            result = await self.mixed.start('instruction', AsyncMock())
            self.assertEqual([v['number'] for v in result['variants']], [3, 4])
            with patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['local1', 'local2'])):
                await self.mixed.retry(result['id'], 'local', AsyncMock())
            api.assert_awaited_once()
        self.assertEqual(len(get_generation(self.ws, result['id'])['variants']), 4)

    async def test_edited_source_blocks_selection_and_retry(self):
        async def api(*args, **kw):
            self.messages[0]['text'] = 'edited question'
            return '{"variants":["a","b"]}'
        with patch('mybot.services.mixed_replies.generate_reply', api), patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(side_effect=ValueError('failed'))):
            result = await self.mixed.start('instruction', AsyncMock())
        self.assertTrue(result['stale'])
        self.assertIsNone(card(result)[1])
        with self.assertRaises(ValueError):
            create_draft(self.ws, result['id'], 4)
        with self.assertRaises(ValueError):
            await self.mixed.retry(result['id'], 'local', AsyncMock())

    async def test_persisted_snapshot_survives_new_messages_but_not_source_edits(self):
        self.ws.chat_history.write_text(json.dumps(self.messages[0]) + '\n')
        async def api(*args, **kwargs):
            self.messages.append(dict(message_id=2, sender='Other', text='new question', type='text'))
            self.ws.chat_history.write_text('\n\n'.join(json.dumps(m) for m in self.messages))
            return '{"variants":["a","b"]}'
        with patch('mybot.services.mixed_replies.generate_reply', api), \
             patch('mybot.services.mixed_replies.local_reply.generate', AsyncMock(return_value=['c','d'])):
            result = await self.mixed.start('instruction', AsyncMock())
        self.assertFalse(result['stale'])
        self.assertEqual([m['message_id'] for m in result['context']], [1])
        self.assertIsNotNone(create_draft(self.ws, result['id'], 4))
        self.messages[0]['text'] = 'edited question'
        self.ws.chat_history.write_text('\n\n'.join(json.dumps(m) for m in self.messages))
        self.assertFalse(self.mixed.current(result))

    def test_bad_pair_rejected(self):
        for text in ('{"variants":["x","x"]}', '{"action":"wait","variants":[]}', 'broken'):
            with self.assertRaises(ValueError):
                parse_pair(text)

    def test_stale_card_preserves_completed_api_text_without_selection(self):
        text, markup = card({'stale': True, 'variants': [{'number': 3, 'source': 'api', 'text': 'already paid answer'}]})
        self.assertIn('already paid answer', text)
        self.assertIsNone(markup)

    async def test_first_local_variant_is_published_before_second(self):
        published = []
        async def publish(record):
            published.append([v['number'] for v in record['variants']])
        async def local(request, on_variant=None):
            await on_variant(['first local'])
            self.assertTrue(any(1 in numbers and 2 not in numbers for numbers in published))
            await on_variant(['first local', 'second local'])
            return ['first local', 'second local']
        with patch('mybot.services.mixed_replies.local_reply.generate', side_effect=local), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["api1","api2"]}')):
            record = await self.mixed.start('instruction', publish)
        self.assertEqual([v['number'] for v in record['variants']], [1, 2, 3, 4])
        self.assertEqual(record['sides']['local']['status'], 'ready')

    async def test_cancel_does_not_publish_outdated_lesson_in_finally(self):
        entered = asyncio.Event()
        async def local(request, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        async def publish(record):
            self.assertFalse(record.get('cancelled'), 'Cancellation must not redraw an old lesson.')
        with patch('mybot.services.mixed_replies.local_reply.generate', side_effect=local), \
             patch('mybot.services.mixed_replies.generate_reply', AsyncMock(return_value='{"variants":["a","b"]}')):
            task = asyncio.create_task(self.mixed.start('instruction', publish))
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_late_results_do_not_reopen_selected_card(self):
        from mybot.telegram.bot_mixed_replies import _generate
        sent = SimpleNamespace(edit_text=AsyncMock())
        sender = AsyncMock(return_value=sent)
        bot = SimpleNamespace(workspace=self.ws, owner_id=1,
            application=SimpleNamespace(bot=SimpleNamespace(send_message=sender)),
            state=SimpleNamespace(current_mood=None))
        self.service.apply_mood = lambda text, mood: text
        chosen = False
        async def start(instruction, publish):
            nonlocal chosen
            record = {'id': 'test', 'variants': [], 'sides': {}, 'preparation': {}}
            await publish(record)
            chosen = True
            await publish(record)
        with patch('mybot.telegram.bot_mixed_replies.MixedReplies') as generator, \
             patch('mybot.telegram.bot_mixed_replies.card', return_value=('variants', None)), \
             patch('mybot.storage.reply_journal.generation_has_draft', side_effect=lambda *args: chosen):
            generator.return_value.start = start
            await _generate(bot, SimpleNamespace(effective_message=None), 'instruction', self.service)
        sender.assert_awaited_once()
        sent.edit_text.assert_not_awaited()

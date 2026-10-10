"""Offline contract, provider, evidence and non-reply UI checks."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.ai.reply_contract import REPLY_INSTRUCTIONS, VARIANT_INSTRUCTIONS, parse_decision
from mybot.ai.reply_provider import generate_reply
from mybot.config import Config
from mybot.services.reply_service import ReplyService
from mybot.storage.workspace import Workspace
from mybot.storage.reply_journal import (
    get_generation_evidence, record_generation, review_counts, review_generation,
    create_draft, mark_draft_kept, record_outgoing_reply, link_outgoing_to_draft,
    finalize_draft_feedback, finalized_correction_candidates,
)
from mybot.telegram.bot_interface import BotInterface
from mybot.app.session_state import SessionState
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_reply_review import review_markup, handle_review


def decision(action='reply'):
    return {'action': action, 'tactic': 'Уточнить планы', 'owner_note': 'Вы свободны завтра?',
            'variants': ['a', 'b', 'c'] if action == 'reply' else []}


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    def test_local_live_prompt_ends_with_current_partner_turn(self):
        from mybot.ai.local_reply import live_messages
        result = live_messages('Historical reply: good morning', [
            {'message_id': 1, 'sender': 'Я', 'text': 'My previous reply'},
            {'message_id': 2, 'sender': 'Other', 'text': 'My current question'}])
        self.assertEqual(result[-1], {'role': 'user', 'content': 'My current question'})
        self.assertEqual(result[-2], {'role': 'assistant', 'content': 'My previous reply'})
        self.assertIn('Historical reply', result[0]['content'])

    async def test_reply_context_is_chronological_and_stale_result_is_rejected(self):
        recent = [{'message_id': 3, 'text': 'Latest', 'sender': 'Other'},
                  {'message_id': 2, 'text': 'Earlier', 'sender': 'Я'}]
        service = ReplyService(recent, {}, self.ws)

        async def generate(*args, **kwargs):
            recent.append({'message_id': 4, 'text': 'New arrival', 'sender': 'Other'})
            return json.dumps(decision())

        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='context')) as build, \
             patch('mybot.services.reply_service.generate_answers', side_effect=generate):
            with self.assertRaisesRegex(ValueError, 'Переписка изменилась'):
                await service.generate()
        self.assertEqual([m['message_id'] for m in build.await_args.args[0]], [2, 3])
        self.assertEqual(service.last_trigger_text, 'Latest')
        self.assertIsNone(service.last_request_snapshot)

    async def test_reply_takes_snapshot_after_waiting_for_lock(self):
        recent = [{'message_id': 1, 'text': 'Old', 'sender': 'Other'}]
        service = ReplyService(recent, {}, self.ws)
        await service.generation_lock.acquire()
        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='context')) as build, \
             patch('mybot.services.reply_service.generate_answers', AsyncMock(return_value=json.dumps(decision()))):
            task = asyncio.create_task(service.generate())
            await asyncio.sleep(0)
            recent.append({'message_id': 2, 'text': 'Latest', 'sender': 'Other'})
            service.generation_lock.release()
            await task
        self.assertEqual([m['message_id'] for m in build.await_args.args[0]], [1, 2])

    async def test_local_wait_uses_selected_adapter_and_force_generates_variants(self):
        service = ReplyService([{'message_id': 3, 'text': 'Ciao', 'sender': 'Other'}], {}, self.ws)
        def snapshot(prompt, workspace, version, **kwargs):
            return {'provider': 'mlx', 'version': version, 'model': 'local',
                    'messages': kwargs.get('messages', []), 'count': kwargs.get('count', 3)}
        with patch('mybot.services.reply_service.selected', return_value='a' * 32), \
             patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='context')), \
             patch('mybot.ai.local_reply.snapshot', side_effect=snapshot), \
             patch('mybot.ai.local_reply.decide', AsyncMock(return_value='wait')) as decide, \
             patch('mybot.ai.local_reply.generate', AsyncMock(return_value=['a', 'b', 'c'])) as generate:
            result = await service.generate(allow_decision=True)
            self.assertEqual(parse_decision(result)['action'], 'wait')
            generate.assert_not_awaited()
            self.assertEqual(decide.await_args.args[0]['version'], 'a' * 32)
            self.assertEqual(service.last_request_snapshot['decision_request']['count'], 1)
            forced = await service.generate(allow_decision=False, force_reply=True)
            self.assertEqual(ReplyService.split_variants(forced), ['a', 'b', 'c'])
            self.assertEqual(generate.await_args.args[0]['version'], 'a' * 32)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.ws = Workspace(1, 10, 'Test', Path(temp.name) / 'one')
        self.other = Workspace(2, 10, 'Other', Path(temp.name) / 'two')

    async def test_contract_and_legacy(self):
        for action in ['reply', 'wait', 'ask_owner']:
            data = decision(action)
            self.assertEqual(parse_decision(json.dumps(data)), data)
            self.assertEqual(ReplyService.split_variants(json.dumps(data)), data['variants'])
        self.assertIsNone(parse_decision('{"variants":["a","b","c"]}'))
        self.assertEqual(ReplyService.split_variants('1. a\n2. b\n3. c'), ['a', 'b', 'c'])
        for data in [dict(decision(), action='send'), dict(decision(), action=[]),
                     dict(decision(), variants=['a']), dict(decision(), variants=['a'] * 3),
                     dict(decision('wait'), variants=['unexpected']),
                     dict(decision('ask_owner'), owner_note='')]:
            with self.assertRaises(ValueError):
                parse_decision(json.dumps(data))
        with self.assertRaises(ValueError):
            ReplyService.split_variants('{"action":"wait",\n bad\n data')

    async def test_service_preview_matches_request_and_failed_run_clears_metadata(self):
        service = ReplyService([{'message_id': 3, 'text': 'Hi', 'sender': 'Other'}], {}, self.ws)
        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(return_value='context')), \
             patch('mybot.services.reply_service.generate_answers', AsyncMock(return_value=json.dumps(decision('wait')))):
            await service.generate(allow_decision=True)
        snapshot = json.loads(self.ws.ai_request_preview.read_text())
        self.assertEqual(snapshot, service.last_request_snapshot)
        self.assertEqual(snapshot['instructions'], REPLY_INSTRUCTIONS)
        self.assertEqual(snapshot['context_message_ids'], [3])
        self.assertEqual(service.last_decision['action'], 'wait')
        with patch('mybot.services.reply_service.build_ai_request', AsyncMock(side_effect=ValueError('bad'))):
            with self.assertRaises(ValueError):
                await service.generate()
        self.assertIsNone(service.last_decision)
        self.assertIsNone(service.last_request_snapshot)

    async def test_compatible_backend_uses_separate_key_and_no_fallback(self):
        default = NS(responses=NS(create=AsyncMock()))
        local = NS(chat=NS(completions=NS(create=AsyncMock(return_value=
                    NS(choices=[NS(message=NS(content='result'))])))))
        manager = AsyncMock()
        manager.__aenter__.return_value = local
        with patch.object(Config, 'REPLY_PROVIDER', 'compatible'), \
             patch.object(Config, 'REPLY_BASE_URL', 'http://localhost:8000/v1'), \
             patch.object(Config, 'REPLY_MODEL', 'local-model'), \
             patch.object(Config, 'REPLY_API_KEY', ''), \
             patch('mybot.ai.reply_provider.AsyncOpenAI', return_value=manager) as factory:
            self.assertEqual(await generate_reply('context', default), 'result')
            self.assertEqual(factory.call_args.kwargs['api_key'], 'local-no-key')
            args = local.chat.completions.create.await_args.kwargs
            self.assertEqual(args['model'], 'local-model')
            self.assertEqual(args['messages'][0]['content'], VARIANT_INSTRUCTIONS)
            local.chat.completions.create.side_effect = RuntimeError('offline')
            with self.assertRaises(RuntimeError):
                await generate_reply('context', default)
            default.responses.create.assert_not_awaited()

    async def test_invalid_provider_config_fails_before_network(self):
        default = NS(responses=NS(create=AsyncMock()))
        with patch.object(Config, 'REPLY_PROVIDER', 'compatible'), \
             patch.object(Config, 'REPLY_BASE_URL', ''), \
             patch.object(Config, 'REPLY_MODEL', ''), \
             patch('mybot.ai.reply_provider.AsyncOpenAI') as factory:
            with self.assertRaises(ValueError):
                await generate_reply('context', default)
            factory.assert_not_called()
        with patch.object(Config, 'REPLY_PROVIDER', 'typo'):
            with self.assertRaises(ValueError):
                await generate_reply('context', default)
        default.responses.create.assert_not_awaited()

    async def test_evidence_review_isolation_and_no_inferred_learning(self):
        data = decision()
        key = record_generation(self.ws, data['variants'], 3, 'model', 'Hi',
                                request_snapshot={'input': 'context'}, decision=data)
        self.assertEqual(get_generation_evidence(self.ws, key)['decision'], data)
        self.assertIsNone(get_generation_evidence(self.other, key))
        # Even an accidentally copied database must not expose another account's evidence.
        wrong = Workspace(2, 10, 'Other', self.ws.root)
        self.assertIsNone(get_generation_evidence(wrong, key))
        with self.assertRaises(ValueError):
            review_generation(wrong, key, 'good')
        review_generation(self.ws, key, 'good')
        self.assertEqual(finalized_correction_candidates(self.ws), [])
        draft = create_draft(self.ws, key, 1)
        mark_draft_kept(self.ws, draft)
        record_outgoing_reply(self.ws, {'message_id': 4, 'sender': 'Я', 'text': 'actual'})
        link_outgoing_to_draft(self.ws, 4, draft)
        finalize_draft_feedback(self.ws, draft)
        self.assertEqual(len(finalized_correction_candidates(self.ws)), 1)
        review_generation(self.ws, key, 'tactic')
        self.assertEqual(review_counts(self.ws), {'tactic': 1})
        self.assertEqual(finalized_correction_candidates(self.ws), [])
        self.assertEqual(review_counts(wrong), {})

    async def test_nonreply_ui_records_decision_without_variant_or_send_buttons(self):
        data = decision('ask_owner')
        service = NS(generate=AsyncMock(return_value=json.dumps(data)),
                     split_variants=ReplyService.split_variants, last_decision=data,
                     last_request_snapshot={'input': 'context'}, last_context_message_id=3)
        bot = BotInterface(1, [], service, SessionState(), self.ws, telegram_client=Mock())
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock(), refresh_reply_context=AsyncMock())
        bot.callbacks.show = AsyncMock()
        bot.send_long_text = AsyncMock()
        update = NS(effective_message=Mock(), effective_user=NS(id=1),
                    effective_chat=NS(id=1, type='private'))
        await bot.send_generated_answers(update)
        bot.send_long_text.assert_awaited_once()
        call = bot.send_long_text.await_args
        self.assertIn('Нужно ваше решение', call.args[1])
        buttons = call.kwargs['reply_markup'].inline_keyboard
        key = buttons[0][0].callback_data.split(':')[2]
        self.assertEqual(get_generation_evidence(self.ws, key)['decision'], data)
        self.assertTrue(all(b.callback_data.startswith('ui:rr:') for row in buttons for b in row))
        bot.telegram_client.send_message.assert_not_called()

    async def test_review_card_is_preserved_and_callback_is_owner_guarded(self):
        key = record_generation(self.ws, ['a', 'b', 'c'], 3, 'model')
        markup = review_markup(key)
        self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=markup)))
        telethon_markup = NS(rows=[NS(buttons=[NS(data=b.callback_data.encode()) for b in row])
                                 for row in markup.inline_keyboard])
        self.assertTrue(BotMenu.is_variant_message(NS(reply_markup=telethon_markup)))
        bot = NS(is_owner=lambda update: False, goal=NS(workspace=lambda: self.ws),
                 runtime_controller=NS(lock=asyncio.Lock()), callbacks=NS(show=AsyncMock()))
        await handle_review(bot, f'ui:rr:{key}:good', None)
        self.assertEqual(review_counts(self.ws), {})
        bot.is_owner = lambda update: True
        await handle_review(bot, f'ui:rr:{key}:good', None)
        self.assertEqual(review_counts(self.ws), {'good': 1})
        bot.goal.workspace = lambda: self.other
        await handle_review(bot, f'ui:rr:{key}:facts', None)
        self.assertEqual(review_counts(self.other), {})
        self.assertEqual(review_counts(self.ws), {'good': 1})

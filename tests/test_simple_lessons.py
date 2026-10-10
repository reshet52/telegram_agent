"""Synthetic lessons: hidden targets, source safety, queue evidence and minimal UI."""

import asyncio
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.services.simple_lessons import start, current, confirm, confirmed
from mybot.services.online_learning import read_state
from mybot.storage.atomic import write_json
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_keyboards import main_menu, control_panel
from mybot.telegram.bot_simple_lesson import BotSimpleLesson


class LessonTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_publish_saves_results_without_redrawing_completed_situation(self):
        from mybot.services.simple_lessons import advance
        state = start(self.ws)
        key = state['cases'][0]
        ui = BotSimpleLesson(self.bot())
        ui.screen = (str(self.ws.root), key)
        ui.render = AsyncMock()
        advance(self.ws, state)
        record = {'variants': [], 'sides': {}, 'cancelled': True}
        await ui.publish(self.ws, key, record)
        ui.render.assert_not_awaited()
        self.assertEqual(PracticeStore(self.ws).get(key)['mixed_record'], record)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ws = Workspace(1, 2, 'synthetic', Path(self.temp.name))
        base = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.messages = [dict(message_id=i+1, sender='Other' if i % 2 == 0 else 'Я', type='text',
            text=f'question {i}' if i % 2 == 0 else f'HIDDEN_ANSWER_{i}',
            date=(base + timedelta(hours=(i//2)*5, seconds=i%2)).isoformat()) for i in range(40)]
        self.save()
        self.reserved = patch('mybot.services.simple_lessons.reserved_ids', return_value=set())
        self.reserved.start()
        self.addCleanup(self.reserved.stop)

    def save(self):
        self.ws.chat_history.write_text(''.join(json.dumps(m)+'\n\n' for m in self.messages))

    def bot(self):
        bot = NS(workspace=self.ws, owner_id=1, main_menu=lambda: main_menu(True),
                 application=NS(bot=NS(send_message=AsyncMock())), callbacks=NS(show=AsyncMock()))
        return bot

    def test_menus_have_only_four_actions(self):
        self.assertEqual([b.callback_data for row in main_menu(True).inline_keyboard for b in row],
                         ['ui:reply', 'ui:reply_instruction', 'ui:dialogs:0', 'ui:learn'])
        self.assertEqual([b.text for row in control_panel(False).keyboard for b in row], ['Выбор диалога'])

    async def test_controls_follow_new_answers_and_old_controls_are_removed(self):
        state = start(self.ws)
        key = state['cases'][0]
        bot = self.bot()
        events = []
        panels = []
        async def send(**kwargs):
            events.append(kwargs['text'])
            panel = NS(edit_reply_markup=AsyncMock(), edit_text=AsyncMock())
            panels.append(panel)
            return panel
        bot.application.bot.send_message.side_effect = send
        ui = BotSimpleLesson(bot)
        ui.screen = (str(self.ws.root), key)
        await ui.render(self.ws, key, None)
        old = ui.panel
        record = {'variants': [{'number': 3, 'source': 'api', 'text': 'first API answer'}], 'sides': {}}
        await ui.render(self.ws, key, record)
        old.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
        self.assertIn('first API answer', events[-2])
        self.assertTrue(events[-1].startswith('Ситуация'))
        current_panel = ui.panel
        await ui.render(self.ws, key, record)
        self.assertIs(ui.panel, current_panel)
        self.assertEqual(len(events), 3)

    async def test_lesson_context_hides_dates_and_pause_labels(self):
        state = start(self.ws)
        key = state['cases'][0]
        store = PracticeStore(self.ws)
        case = store.get(key)
        case['mixed_record'] = {'variants': [], 'sides': {}, 'preparation': {}}
        write_json(store.path(key), case)
        bot = self.bot()
        ui = BotSimpleLesson(bot)
        try:
            await ui.show_case(self.ws)
            texts = [c.kwargs['text'] for c in bot.application.bot.send_message.await_args_list]
            context = next(t for t in texts if t.startswith('Учебная ситуация'))
            self.assertIn('Other: question', context)
            self.assertNotIn('2025-', context)
            self.assertNotIn('Пауза', context)
            self.assertNotIn('HIDDEN_ANSWER', context)
        finally:
            ui.leave()
            if ui.watcher:
                await asyncio.gather(ui.watcher, return_exceptions=True)

    async def test_hidden_answer_not_sent_to_either_provider(self):
        state = start(self.ws)
        key = state['cases'][0]
        ui = BotSimpleLesson(self.bot())
        generator = ui.generator(self.ws, key)
        generator.enhanced = False
        async def local(request, **kwargs):
            self.assertNotIn('HIDDEN_ANSWER', json.dumps(request['messages']))
            return ['local1', 'local2']
        async def api(prompt, *args, **kwargs):
            self.assertNotIn('HIDDEN_ANSWER', prompt)
            return '{"variants":["api1","api2"]}'
        with patch('mybot.services.mixed_replies.selected', return_value='base'), patch(
                'mybot.services.mixed_replies.local_reply.snapshot', side_effect=lambda *a, **k: dict(model='local', version='base', **k)), patch(
                'mybot.services.mixed_replies.local_reply.generate', side_effect=local), patch(
                'mybot.services.mixed_replies.generate_reply', side_effect=api), patch(
                'mybot.services.online_learning.pause_for_reply', AsyncMock()):
            record = await generator.start('instruction', AsyncMock())
        self.messages.append(dict(message_id=41, sender='Other', text='new live message', type='text', date='2025-02-01T00:00:00+00:00'))
        self.save()
        self.assertTrue(generator.current(record))
        _, case = current(self.ws, key)
        self.messages[case['context_ids'][0]-1]['text'] = 'edited source'
        self.save()
        self.assertFalse(generator.current(record))

    def test_fourth_choice_and_own_reply_are_durable_distinct_evidence(self):
        state = start(self.ws)
        key = state['cases'][0]
        confirm(self.ws, key, 'chosen API', {'id': 'a'*32}, 4)
        item = read_state(self.ws)['queue'][0]
        self.assertEqual(item['source_kind'], 'owner_confirmed_reply')
        self.assertEqual(item['provenance']['chosen_variant'], 4)
        self.assertEqual(confirmed(self.ws, item)['text'], 'chosen API')
        _, case = current(self.ws, state['cases'][1])
        confirm(self.ws, case['id'], 'my edit')
        self.assertEqual(len(read_state(self.ws)['queue']), 2)

    def test_revision_replaces_pending_and_old_evidence_remains_valid(self):
        state = start(self.ws)
        key = state['cases'][0]
        confirm(self.ws, key, 'first')
        old = read_state(self.ws)['queue'][0]
        StyleLesson(self.ws).write('simple_lesson.json', state)
        confirm(self.ws, key, 'second')
        new = read_state(self.ws)['queue']
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0]['text'], 'second')
        self.assertEqual(confirmed(self.ws, old)['text'], 'first')
        StyleLesson(self.ws).write('simple_lesson.json', state)
        confirm(self.ws, key, 'second')
        self.assertEqual(len(read_state(self.ws)['queue']), 1)

    async def test_skip_and_home_do_not_add_examples_and_cancel_input(self):
        state = start(self.ws)
        key = state['cases'][0]
        state['input_case'] = key
        StyleLesson(self.ws).write('simple_lesson.json', state)
        ui = BotSimpleLesson(self.bot())
        ui.leave()
        self.assertNotIn('input_case', StyleLesson(self.ws).read('simple_lesson.json'))
        self.assertEqual(read_state(self.ws)['queue'], [])
        self.assertEqual(start(self.ws)['cases'], state['cases'])

    def test_reserved_source_cannot_be_confirmed(self):
        state = start(self.ws)
        key = state['cases'][0]
        case = PracticeStore(self.ws).get(key)
        with patch('mybot.services.simple_lessons.reserved_ids', return_value=set(case['expected_ids'])), self.assertRaises(ValueError):
            confirm(self.ws, key, 'wrong')

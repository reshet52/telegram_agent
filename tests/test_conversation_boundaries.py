"""Meaningful session and action examples; no future text enters model prompts."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mybot.services.conversation_boundaries import sessions, wait_review_candidates, crossing_reason
from mybot.services.decision_task import training_row, confirmed_counts
from mybot.services.reviewed_dataset import action_examples
from mybot.storage.reply_practice import PracticeStore, candidates
from mybot.storage.workspace import Workspace
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.atomic import write_json


def message(number, minutes, sender, text):
    return {'message_id': number, 'date': (datetime(2026, 1, 1, tzinfo=timezone.utc)
            + timedelta(minutes=minutes)).isoformat(), 'sender': sender, 'text': text,
            'type': 'text'}


class BoundaryTests(unittest.TestCase):
    def test_overnight_greeting_is_not_a_reply_to_farewell(self):
        history = [message(1, 0, 'Я', 'Good night'), message(2, 1, 'Other', 'Ciao'),
                   message(3, 600, 'Я', 'Good morning'), message(4, 601, 'Other', 'Morning'),
                   message(5, 602, 'Я', 'Hi')]
        pairs = candidates(history, set(), set())
        self.assertEqual([e[0]['message_id'] for _, e in pairs], [5])
        self.assertEqual([m['message_id'] for m in pairs[0][0]], [3, 4])
        waits = wait_review_candidates(history, set())
        self.assertEqual(len(waits), 1)
        self.assertEqual(waits[0][0][-1]['message_id'], 2)
        self.assertEqual(waits[0][1]['next_message_id'], 3)

    def test_first_goodbye_is_not_automatically_a_wait_label(self):
        history = [message(1, 0, 'Other', 'Ciao'), message(2, 1, 'Я', 'Ciao'),
                   message(3, 600, 'Other', 'Good morning')]
        self.assertEqual(len(candidates(history, set(), set())), 1)
        self.assertEqual(wait_review_candidates(history, set()), [])
        self.assertEqual(len(sessions(history)), 2)
        self.assertIsNone(crossing_reason(history[:1], history[1:2]))

    def test_confirmed_pause_is_an_action_target_without_future_text(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(7, 9, 'test', Path(directory))
            history = [message(1, 0, 'Я', 'Good night'), message(2, 1, 'Other', 'Ciao'),
                       message(3, 600, 'Other', 'Good morning')]
            ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in history))
            store = PracticeStore(ws)
            case = store.create_from(history[:2], [], 'wait_review')
            self.assertEqual(action_examples(ws), [])
            store.confirm_action(case['id'], 'wait')
            items = action_examples(ws)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]['row'], training_row(history[:2], 'wait'))
            self.assertNotIn('Good morning', json.dumps(items[0]['row']['prompt']))
            self.assertEqual(items[0]['row']['completion'][0]['content'], 'WAIT')
            store.save_decision(case['id'], 'reply', {'version': 'a' * 32})
            self.assertEqual(confirmed_counts(ws), {'reviewed': 1, 'wrong_reply': 1, 'wrong_wait': 0})

    def test_old_holdout_is_kept_when_replacement_is_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            ws = Workspace(7, 9, 'test', Path(directory))
            history = [message(i, i, 'Other' if i % 2 else 'Я', f'text {i}') for i in range(1, 101)]
            history[50]['date'] = message(51, 1000, 'Other', 'x')['date']
            for i in range(51, 100):
                history[i]['date'] = message(i + 1, 1000 + i, 'Other', 'x')['date']
            ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in history))
            lesson = StyleLesson(ws)
            old = lesson.store.create_from(history[47:50], history[50:51], 'holdout')
            lesson.write('holdout.json', {'version': 1, 'cases': [old['id']]})
            self.assertTrue(lesson.holdout_audit()[0]['reason'])
            with self.assertRaises(ValueError):
                lesson.replace_holdout()
            preview = lesson.save_replacement_preview()
            self.assertGreaterEqual(len(preview), 2)
            replacement = lesson.replace_holdout()
            self.assertNotEqual(replacement['cases'], [old['id']])
            self.assertEqual(lesson.holdout_for('holdout.json')['cases'], [old['id']])
            self.assertTrue(all(not item['reason'] for item in lesson.holdout_audit()))

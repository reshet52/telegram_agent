import json
import tempfile
import unittest
from pathlib import Path

from mybot.services.conversation_timing import context_timing, display_context, history_report
from mybot.services.simple_lessons import start, confirm_wait
from mybot.services.online_learning import read_state
from mybot.storage.reply_practice import PracticeStore
from mybot.storage.workspace import Workspace
from mybot.storage.atomic import read_json, write_json


def message(i, date, sender='Other', text='good night'):
    return dict(message_id=i, date=date, sender=sender, text=text, type='text')


class TimingTests(unittest.TestCase):
    def test_offsets_gaps_missing_dates_and_no_future_reply(self):
        messages = [message(1, '2026-01-01T23:00:00+03:00'),
                    message(2, '2026-01-01T20:10:00Z')]
        value = context_timing(messages, reply_time=messages[-1]['date'])
        self.assertEqual(value['observed_duration_seconds'], 600)
        self.assertEqual(value['seconds_since_last_message'], 0)
        self.assertIn('Пауза: 10 мин.', display_context(messages))
        self.assertIsNone(context_timing([message(3, None)])['messages'][0]['sent_at'])
        self.assertIsNone(context_timing([message(3, '2026-01-01T20:10:00')])['messages'][0]['sent_at'])

    def test_report_updates_without_rewriting_history_and_refuses_foreign_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            messages = [message(1, '2026-01-01T20:00:00Z'),
                        message(2, '2026-01-01T20:01:00Z', 'Я'),
                        message(3, '2026-01-02T08:00:00Z')]
            original = ''.join(json.dumps(m) + '\n\n' for m in messages)
            ws.chat_history.write_text(original)
            result = history_report(ws)
            self.assertEqual(len(result['sessions']), 2)
            self.assertEqual(result['median_owner_reply_delay_seconds'], 60)
            self.assertEqual(ws.chat_history.read_text(), original)
            self.assertEqual(history_report(ws), result)
            result['dialog_id'] = 3
            write_json(ws.root / 'reply_assistant/timing_report.json', result)
            with self.assertRaises(ValueError):
                history_report(ws)

    def test_owner_wait_is_saved_separately_not_an_empty_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp))
            messages = [message(1, '2026-01-01T20:00:00Z'),
                        message(2, '2026-01-01T20:01:00Z', 'Я')]
            ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
            state = start(ws)
            key = state['cases'][0]
            confirm_wait(ws, key)
            self.assertEqual(PracticeStore(ws).get(key)['owner_action']['action'], 'wait')
            self.assertEqual(read_state(ws)['queue'], [])
            self.assertEqual(read_json(ws.root / 'learning/simple_lesson.json')['choices'][key], 'wait')


if __name__ == '__main__':
    unittest.main()

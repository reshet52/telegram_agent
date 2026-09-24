"""Only explicit, same-dialog corrections may enter the generation request."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from mybot.ai.context_builder import build_ai_request
from mybot.services.correction_examples import relevant_correction_examples
from mybot.services.reply_service import ReplyService
from mybot.storage.reply_journal import (
    create_draft, finalize_draft_feedback, link_outgoing_to_draft,
    mark_draft_kept, record_generation, record_outgoing_reply)
from mybot.storage.workspace import Workspace
from mybot.storage.deletions import mark_messages_deleted


class CorrectionExampleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.workspace = Workspace(1, 42, 'Anna', root / 'account_1/dialogs/dialog_42')
        self.other = Workspace(1, 13, 'Other', root / 'account_1/dialogs/dialog_13')

    def feedback(self, workspace, message_id, trigger, variant, actual, finalize=True):
        generation = record_generation(workspace, [variant, 'second', 'third'],
                                       message_id - 1, 'gpt-6-luna', trigger)
        draft = create_draft(workspace, generation, 1)
        mark_draft_kept(workspace, draft)
        record_outgoing_reply(workspace, {'message_id': message_id, 'sender': 'Я',
                                          'text': actual, 'type': 'text'})
        link_outgoing_to_draft(workspace, message_id, draft)
        if finalize:
            finalize_draft_feedback(workspace, draft)
        return generation

    async def test_retrieval_requires_finalization_overlap_and_same_workspace(self):
        wanted = self.feedback(self.workspace, 101,
            'How was your day at the beach?', 'A long poetic beach reply',
            'Lovely beach day ❤️')
        self.feedback(self.workspace, 102, 'Where is the train?',
                      'Train answer', 'The train is late')
        self.feedback(self.workspace, 103, 'How was your beach vacation?',
                      'pending suggestion', 'pending reply', finalize=False)
        self.feedback(self.other, 101, 'How was your beach vacation?',
                      'private other-person fact', 'private actual reply')
        examples = relevant_correction_examples(self.workspace, 'How was the beach?',
                                                max_chars=1800)
        self.assertEqual([item['generation_id'] for item in examples], [wanted])
        self.assertEqual(examples[0]['actual_reply'], 'Lovely beach day ❤️')
        self.assertEqual(examples[0]['result'], 'accepted_with_edit')
        self.assertEqual(relevant_correction_examples(self.other, 'Where is the train?'), [])
        self.assertEqual(relevant_correction_examples(self.workspace, 'completely unrelated'), [])

    async def test_prompt_contains_bounded_source_and_can_disable_examples(self):
        generation = self.feedback(self.workspace, 201,
            'How was your beach day?', 'Long AI answer', 'Short real answer')
        self.feedback(self.other, 201, 'How was your beach day?',
                      'private other-person fact', 'private actual reply')
        messages = [{'sender': 'Anna', 'text': 'How was your beach day?',
                     'message_id': 300, 'type': 'text'}]
        with patch('mybot.ai.context_builder.load_global_style', return_value={}), \
             patch('mybot.ai.context_builder.find_relevant_memories',
                   new_callable=AsyncMock, return_value=[]), \
             patch('mybot.ai.context_builder.find_similar_episodes',
                   new_callable=AsyncMock, return_value=[]):
            prompt = await build_ai_request(messages, 'reply', {},
                workspace=self.workspace, use_correction_examples=True)
            disabled = await build_ai_request(messages, 'reply', {},
                workspace=self.workspace, use_correction_examples=False)
        self.assertIn(generation, prompt)
        self.assertIn('Short real answer', prompt)
        self.assertNotIn('private other-person fact', prompt)
        self.assertNotIn('Long AI answer', disabled)
        self.assertLessEqual(len(str(relevant_correction_examples(
            self.workspace, 'beach day', max_chars=600))), 700)

    async def test_reply_service_snapshots_trigger_before_generation(self):
        messages = [{'sender': 'Anna', 'text': 'How was your day?',
                     'message_id': 10, 'type': 'text'}]
        service = ReplyService(messages, {}, workspace=self.workspace)
        with patch('mybot.services.reply_service.build_ai_request',
                   new_callable=AsyncMock, return_value='request'), \
             patch('mybot.services.reply_service.generate_answers',
                   new_callable=AsyncMock, return_value='response'):
            await service.generate()
        messages.append({'sender': 'Anna', 'text': 'Later message',
                         'message_id': 11, 'type': 'text'})
        self.assertEqual(service.last_trigger_text, 'How was your day?')
        self.assertEqual(service.last_context_message_id, 10)

    async def test_deleted_outgoing_is_not_used_as_example(self):
        self.feedback(self.workspace, 401, 'A distinct beach question',
                      'AI text', 'Real reply')
        self.assertEqual(len(relevant_correction_examples(
            self.workspace, 'beach question')), 1)
        mark_messages_deleted({401}, filename=self.workspace.deleted_message_ids)
        self.assertEqual(relevant_correction_examples(
            self.workspace, 'beach question'), [])


if __name__ == '__main__':
    unittest.main()

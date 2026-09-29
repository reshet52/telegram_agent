"""Offline semantic search and UI isolation checks; no external API calls."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.app.session_state import SessionState
from mybot.services.history_search import _semantic_material, _rank_evidence, search_history
from mybot.storage.deletions import mark_messages_deleted
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_interface import BotInterface


class HistorySearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.workspace = Workspace(1, 42, "Anna", root / "account_1/dialogs/dialog_42")
        self.other = Workspace(2, 42, "Other", root / "account_2/dialogs/dialog_42")
        self.workspace.root.mkdir(parents=True)
        self.other.root.mkdir(parents=True)

    def save(self, workspace, records):
        workspace.chat_history.write_text("".join(
            json.dumps(item, ensure_ascii=False, indent=4) + "\n\n" for item in records),
            encoding="utf-8")

    def episode(self, workspace, record, embedding):
        workspace.episodes.write_text(json.dumps({"episode_id": 1, "previous_context": [],
            "incoming": [record], "response": []}, ensure_ascii=False) + "\n")
        workspace.episode_embeddings.write_text(json.dumps({"episodes": [
            {"episode_id": 1, "embedding": embedding}]}))

    def test_semantic_retrieval_finds_unrelated_wording_past_500_and_excludes_deleted(self):
        records = [{"message_id": number, "date": "2024-01-01T10:00:00+00:00",
                    "sender": "Я", "text": "ordinary conversation"}
                   for number in range(1, 501)]
        fact = {"message_id": 501, "date": "2024-03-02T10:01:00+00:00",
                "sender": "Anna", "text": "Rex chewed the sofa again."}
        records.extend([fact, {"message_id": 502, "sender": "Anna", "text": "wrong deleted"}])
        self.save(self.workspace, records)
        self.episode(self.workspace, fact, [1.0, 0.0])
        mark_messages_deleted([502], filename=self.workspace.deleted_message_ids)
        self.save(self.other, [{"message_id": 501, "sender": "Other",
                                "text": "Different account secret"}])
        material = _semantic_material(self.workspace)
        self.assertEqual(len(material[4]), 500)  # An incomplete index must be reported.
        evidence = _rank_evidence(material, [1.0, 0.0], [])
        self.assertIn(501, {item["message_id"] for item in evidence})
        self.assertNotIn(502, {item["message_id"] for item in evidence})
        self.assertNotIn("Different account secret", json.dumps(evidence))

    async def test_small_unindexed_tail_is_cached_and_cited(self):
        fact = {"message_id": 1, "date": "2024-03-02T10:00:00+00:00",
                "sender": "Anna", "text": "Rex chewed the sofa again."}
        self.save(self.workspace, [fact])
        embedding = NS(data=[NS(embedding=[1.0, 0.0])])
        answer = NS(output_text=json.dumps({"candidates": [{"answer": "Его зовут Рекс.",
                                           "source_ids": [999, 1]}]}, ensure_ascii=False))
        with patch("mybot.services.history_search.ai_client.embeddings.create",
                   new=AsyncMock(return_value=embedding)) as embed, \
             patch("mybot.services.history_search.create_query_embedding",
                   new=AsyncMock(return_value=[1.0, 0.0])), \
             patch("mybot.services.history_search.ai_client.responses.create",
                   new=AsyncMock(return_value=answer)) as respond:
            first = await search_history(self.workspace, "Как зовут собаку?")
            second = await search_history(self.workspace, "Как зовут собаку?")
        self.assertIn("Рекс", first)
        self.assertIn("2024-03-02 10:00", first)
        self.assertIn("ID 1", first)
        self.assertNotIn("ID 999", first)
        self.assertEqual(first, second)
        embed.assert_awaited_once()
        self.assertEqual(respond.await_count, 2)
        self.assertTrue(self.workspace.root.joinpath("history_search_tail.json").exists())

    def test_memory_vector_can_find_an_episode_with_different_wording(self):
        records = [{"message_id": number, "sender": "Anna", "text": "A generic topic"}
                   for number in range(1, 40)]
        fact = {"message_id": 40, "sender": "Anna", "text": "Rex chewed the sofa again."}
        self.save(self.workspace, [*records, fact])
        indexed = [(number, records[number - 1]) for number in range(1, 16)] + [(16, fact)]
        self.workspace.episodes.write_text("".join(json.dumps({
            "episode_id": number, "previous_context": [], "incoming": [message],
            "response": []}) + "\n" for number, message in indexed))
        self.workspace.episode_embeddings.write_text(json.dumps({"episodes": [
            {"episode_id": number, "embedding": [0.0, 1.0] if number == 16
             else [1.0, 0.0]} for number, _ in indexed]}))
        self.workspace.memory_embeddings.write_text(json.dumps({"memories": [
            {"category": "person_facts", "memory": "the pet is called Rex",
             "embedding": [0.3, 0.95]}]}))
        material = _semantic_material(self.workspace)
        with patch("mybot.services.history_search.search_memories", return_value=[]):
            without_hint = _rank_evidence(material, [1.0, 0.0], [])
        with_hint = _rank_evidence(material, [1.0, 0.0], [])
        self.assertNotIn(40, {item["message_id"] for item in without_hint})
        self.assertIn(40, {item["message_id"] for item in with_hint})

    async def test_large_missing_index_does_not_start_bulk_paid_work(self):
        self.save(self.workspace, [{"message_id": index, "text": "hello"}
                                   for index in range(1, 67)])
        with patch("mybot.services.history_search.ai_client.embeddings.create",
                   new=AsyncMock()) as embed, \
             patch("mybot.services.history_search.create_query_embedding",
                   new=AsyncMock()) as query:
            with self.assertRaisesRegex(ValueError, "не покрывает"):
                await search_history(self.workspace, "Что обсуждали?")
        embed.assert_not_awaited()
        query.assert_not_awaited()

    async def test_model_cannot_cite_unseen_message(self):
        self.save(self.workspace, [{"message_id": 1, "text": "hello"}])
        from mybot.services.history_search import answer_from_evidence
        with patch("mybot.services.history_search.ai_client.responses.create",
                   new=AsyncMock(return_value=NS(output_text=json.dumps(
                       {"candidates": [{"answer": "Unsupported", "source_ids": [999]}]})))):
            text = await answer_from_evidence("вопрос", [{"message_id": 1,
                "date": None, "sender": "X", "text": "hello"}])
        self.assertIn("не могу подтвердить", text)

    async def test_button_routes_next_text_to_search_and_cancel_clears_it(self):
        bot = BotInterface(1, [], object(), SessionState(), workspace=self.workspace)
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.menu = NS(show=AsyncMock())
        bot.send_generated_answers = AsyncMock()
        bot.answer_history_question = AsyncMock()
        source = NS(text="Как зовут собаку?", reply_text=AsyncMock())
        update = NS(effective_message=source, effective_user=NS(id=1),
                    effective_chat=NS(id=1, type="private"))
        context = NS(user_data={"pending_reply_edit": (42, "old-draft", 10)})
        await bot.callbacks.dispatch("ui:history_search", update, context)
        self.assertEqual(context.user_data["awaiting_history_search"], 42)
        self.assertNotIn("pending_reply_edit", context.user_data)
        await bot.text_instruction(update, context)
        bot.answer_history_question.assert_awaited_once_with(update, source.text)
        bot.send_generated_answers.assert_not_awaited()
        await bot.callbacks.dispatch("ui:history_search", update, context)
        await bot.callbacks.dispatch("ui:home", update, context)
        self.assertNotIn("awaiting_history_search", context.user_data)

    async def test_plain_text_requires_explicit_reply_action(self):
        bot = BotInterface(1, [], object(), SessionState(), workspace=self.workspace)
        bot.runtime_controller = NS(busy=False, lock=asyncio.Lock())
        bot.menu = NS(show=AsyncMock(), move_to_bottom=AsyncMock())
        bot.send_generated_answers = AsyncMock()
        bot.reply_actions.handle_text = AsyncMock(return_value=False)
        source = NS(text="Как зовут собаку?", reply_text=AsyncMock())
        update = NS(effective_message=source, effective_user=NS(id=1),
                    effective_chat=NS(id=1, type="private"))
        context = NS(user_data={})
        await bot.text_instruction(update, context)
        bot.send_generated_answers.assert_not_awaited()
        self.assertIn("нажмите", bot.menu.show.await_args.args[1])
        await bot.callbacks.dispatch("ui:reply_instruction", update, context)
        self.assertEqual(context.user_data["awaiting_reply_instruction"], 42)
        source.text = "Ответь кратко"
        await bot.text_instruction(update, context)
        bot.send_generated_answers.assert_awaited_once_with(update, "Ответь кратко")
        self.assertNotIn("awaiting_reply_instruction", context.user_data)

    async def test_three_source_backed_candidates_without_duplicate(self):
        from mybot.services.history_search import answer_from_evidence
        evidence = [{"message_id": index, "date": "2024-01-01", "sender": "A",
                     "text": f"source {index}"} for index in range(1, 4)]
        payload = {"candidates": [
            {"answer": "Торонто", "source_ids": [1]},
            {"answer": "Лондон", "source_ids": [2]},
            {"answer": "Лондон", "source_ids": [2]},
            {"answer": "Париж", "source_ids": [3]}]}
        with patch("mybot.services.history_search.ai_client.responses.create",
                   new=AsyncMock(return_value=NS(output_text=json.dumps(payload)))):
            result = await answer_from_evidence("Где?", evidence)
        self.assertIn("Возможный ответ 1: Торонто", result)
        self.assertIn("Возможный ответ 2: Лондон", result)
        self.assertIn("Возможный ответ 3: Париж", result)

    async def test_search_expands_evidence_after_unconfirmed_first_pass(self):
        from mybot.services import history_search as search
        fact = {"message_id": 1, "date": "2024-01-01", "sender": "Anna",
                "text": "I live in Toronto"}
        self.save(self.workspace, [fact])
        first = [{"message_id": 2, "text": "irrelevant"}]
        wider = [{"message_id": 1, "date": "2024-01-01", "sender": "Anna",
                  "text": "I live in Toronto"}]
        with patch.object(search, "_semantic_material", return_value=([fact], {}, {}, {}, [])), \
             patch.object(search, "_tail_vectors", new=AsyncMock(return_value=[])), \
             patch.object(search, "create_query_embedding", new=AsyncMock(return_value=[1.0])), \
             patch.object(search, "_rank_evidence", side_effect=[first, wider]) as rank, \
             patch.object(search, "answer_from_evidence", new=AsyncMock(side_effect=[
                 "По найденным сообщениям я не могу подтвердить ответ. Попробуйте уточнить вопрос.",
                 "Возможный ответ 1: Торонто"])) as answer:
            result = await search.search_history(self.workspace, "Где живёт Anna?")
        self.assertIn("Торонто", result)
        self.assertEqual(rank.call_count, 2)
        self.assertEqual(answer.await_count, 2)


if __name__ == "__main__":
    unittest.main()

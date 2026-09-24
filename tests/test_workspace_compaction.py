"""Offline history finalization: no Telegram or OpenAI requests."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.services.history_import import commit_batch
from mybot.services.runtime_controller import RuntimeController
from mybot.services.workspace_compaction import compact_workspace
from mybot.storage.atomic import read_json, write_json
from mybot.storage.control_state import ControlState
from mybot.storage.history import load_all_messages
from mybot.storage.workspace import create_workspace


def write_lines(path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        patcher = patch("mybot.storage.workspace.DATA_ROOT", Path(temp.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ws = create_workspace(101, 201, "First")
        self.other = create_workspace(102, 201, "Other account")

    def populate(self):
        commit_batch(self.ws.chat_history, [{"message_id": 1, "sender": "Я", "text": "hello"}])
        write_json(self.ws.episode_embeddings, {"model": "test", "episode_count": 2,
            "total_tokens": 5, "episodes": [
                {"episode_id": 1, "embedding": [1]}, {"episode_id": 2, "embedding": [2]}]})
        write_lines(self.ws.episode_embeddings_live, [
            {"episode_id": 2, "embedding": [22]}, {"episode_id": 3, "embedding": [3]}])
        write_json(self.ws.agent_memory, {"person_facts": [{"memory": "old"}]})
        write_lines(self.ws.agent_memory_live, [
            {"category": "person_facts", "memory": "new"}])
        write_json(self.ws.memory_embeddings, {"model": "test", "count": 1, "memories": [
            {"category": "person_facts", "memory": "old", "embedding": [1]}]})
        write_lines(self.ws.memory_embeddings_live, [
            {"category": "person_facts", "memory": "new", "embedding": [2]}])

    def test_merges_once_keeps_raw_history_and_other_account_untouched(self):
        self.populate()
        original_history = self.ws.chat_history.read_bytes()
        write_json(self.other.episode_embeddings, {"episodes": [
            {"episode_id": 99, "embedding": [99]}]})
        other_bytes = self.other.episode_embeddings.read_bytes()
        result = compact_workspace(self.ws)
        self.assertEqual(result, {"episode_embeddings": 2, "memories": 1})
        episodes = read_json(self.ws.episode_embeddings)
        self.assertEqual([item["episode_id"] for item in episodes["episodes"]], [1, 2, 3])
        self.assertEqual(episodes["episodes"][1]["embedding"], [22])
        self.assertEqual(episodes["episode_count"], 3)
        self.assertEqual(read_json(self.ws.agent_memory)["person_facts"][-1]["memory"], "new")
        self.assertEqual(read_json(self.ws.memory_embeddings)["count"], 2)
        self.assertEqual(self.ws.chat_history.read_bytes(), original_history)
        self.assertEqual(self.other.episode_embeddings.read_bytes(), other_bytes)
        self.assertFalse(self.ws.episode_embeddings_live.read_text())
        self.assertFalse(self.ws.agent_memory_live.read_text())
        self.assertFalse(self.ws.memory_embeddings_live.read_text())
        self.assertEqual(compact_workspace(self.ws), {"episode_embeddings": 0, "memories": 0})

    def test_failed_base_write_keeps_live_source_for_retry(self):
        self.populate()
        source = self.ws.episode_embeddings_live.read_bytes()
        with patch("mybot.services.workspace_compaction.write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                compact_workspace(self.ws)
        self.assertEqual(self.ws.episode_embeddings_live.read_bytes(), source)
        self.assertEqual(compact_workspace(self.ws)["episode_embeddings"], 2)

    def test_unpaired_live_memory_is_not_cleared(self):
        self.populate()
        write_lines(self.ws.memory_embeddings_live, [])
        with self.assertRaisesRegex(ValueError, "не имеет соответствующего embedding"):
            compact_workspace(self.ws)
        self.assertTrue(self.ws.agent_memory_live.read_text())
        self.assertEqual(read_json(self.ws.agent_memory)["person_facts"], [{"memory": "old"}])

    def test_partial_memory_publish_retries_without_duplicate_records(self):
        self.populate()
        from mybot.services import workspace_compaction
        original_write = workspace_compaction.write_json
        def interrupted(path, value):
            if path == self.ws.memory_embeddings:
                raise OSError("interrupted before memory index")
            return original_write(path, value)
        with patch("mybot.services.workspace_compaction.write_json", side_effect=interrupted):
            with self.assertRaises(OSError):
                compact_workspace(self.ws)
        self.assertTrue(self.ws.agent_memory_live.read_text())
        self.assertTrue(self.ws.memory_embeddings_live.read_text())
        compact_workspace(self.ws)
        memories = read_json(self.ws.agent_memory)["person_facts"]
        self.assertEqual([item["memory"] for item in memories], ["old", "new"])
        self.assertEqual(read_json(self.ws.memory_embeddings)["count"], 2)

    async def test_finish_button_syncs_history_and_closes_dialog(self):
        self.populate()
        state = ControlState(101)
        state.set(active_dialog_id=201)
        dialog = NS(id=201, name="First")
        listener = NS(selected_dialog=dialog, unregister=AsyncMock(), register=Mock())
        bot = NS(control_state=state, agent_manager=NS(selected_dialog=dialog),
                 application=None, workspace=self.ws, recent_messages=[], reply_service=object())
        controller = RuntimeController(Mock(), NS(id=101), bot)
        controller.listener = listener
        controller.runtime = NS(workspace=self.ws, raw_history=[], recent_messages=[],
                                known_message_ids={1})
        report = AsyncMock()
        with patch("mybot.services.runtime_controller.fetch_messages_after",
                   new=AsyncMock(return_value=[{"message_id": 2, "sender": "Other", "text": "bye"}])):
            self.assertTrue(await controller.finish_dialog(report))
            await controller.finish_task
        self.assertEqual([m["message_id"] for m in load_all_messages(
            filename=self.ws.chat_history)], [1, 2])
        self.assertIsNone(ControlState(101).get("active_dialog_id"))
        self.assertIsNone(controller.listener)
        self.assertIsNone(bot.workspace)
        self.assertFalse(self.ws.episode_embeddings_live.read_text())
        self.assertTrue(report.await_args_list[-1].args[1])

    async def test_failed_finish_keeps_dialog_and_listener_for_retry(self):
        self.populate()
        state = ControlState(101)
        state.set(active_dialog_id=201)
        dialog = NS(id=201, name="First")
        listener = NS(selected_dialog=dialog, unregister=AsyncMock(), register=Mock())
        bot = NS(control_state=state, agent_manager=NS(selected_dialog=dialog),
                 application=None, workspace=self.ws, recent_messages=[], reply_service=object())
        controller = RuntimeController(Mock(), NS(id=101), bot)
        controller.listener = listener
        controller.runtime = NS(workspace=self.ws, raw_history=[], recent_messages=[],
                                known_message_ids={1})
        with patch("mybot.services.runtime_controller.fetch_messages_after",
                   new=AsyncMock(return_value=[])), \
             patch("mybot.services.runtime_controller.compact_workspace",
                   side_effect=OSError("disk full")):
            with self.assertLogs(level="ERROR"):
                self.assertTrue(await controller.finish_dialog(AsyncMock()))
                await controller.finish_task
        listener.register.assert_called_once()
        self.assertEqual(ControlState(101).get("active_dialog_id"), 201)
        self.assertIs(controller.listener, listener)
        self.assertTrue(self.ws.episode_embeddings_live.read_text())


if __name__ == "__main__":
    unittest.main()

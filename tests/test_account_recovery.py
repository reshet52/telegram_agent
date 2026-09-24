"""Offline checks for account isolation and restart checkpoints."""

import json
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.services.account_registry import load_accounts
from mybot.storage.control_state import ControlState
from mybot.telegram.account_bot import AccountBot
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_panel import BotPanel
from telegram import InlineKeyboardMarkup
from telegram.error import BadRequest


class AccountRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root_patch = patch("mybot.storage.workspace.DATA_ROOT", Path(self.temp.name))
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_two_distinct_credential_pairs_and_sessions(self):
        root = Path(self.temp.name) / "account_22" / "global"
        root.mkdir(parents=True)
        (root / "telegram_credentials.json").write_text(
            json.dumps({"api_id": 222, "api_hash": "second"}))
        with patch.dict("os.environ", {"ALLOWED_TELEGRAM_IDS": "11,22",
                                    "PRIMARY_TELEGRAM_ID": "11"}):
            with patch("mybot.services.account_registry.Config.API_ID", 111), \
                 patch("mybot.services.account_registry.Config.API_HASH", "first"), \
                 patch("mybot.services.account_registry.Config.SESSION_NAME", "mybot_session"):
                accounts = load_accounts()
        self.assertEqual((accounts[11].api_id, accounts[11].api_hash,
                          accounts[11].session), (111, "first", "mybot_session"))
        self.assertEqual((accounts[22].api_id, accounts[22].api_hash), (222, "second"))
        self.assertNotEqual(accounts[11].session, accounts[22].session)

    def test_atomic_control_state_preserves_previous_value_on_failure(self):
        state = ControlState(11)
        state.set(active_dialog_id=33, panel_hidden=True, menu_message_id=44,
                  panel_message_id=55)
        with patch("mybot.storage.atomic.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                state.set(active_dialog_id=99)
        recovered = ControlState(11)
        self.assertEqual(recovered.get("active_dialog_id"), 33)
        self.assertTrue(recovered.get("panel_hidden"))
        self.assertEqual(recovered.get("menu_message_id"), 44)
        self.assertEqual(recovered.get("panel_message_id"), 55)

    async def test_saved_menu_is_edited_and_hidden_panel_is_restored(self):
        state = ControlState(11)
        state.set(menu_message_id=44, panel_message_id=55, panel_hidden=True)
        api = NS(edit_message_text=AsyncMock(), send_message=AsyncMock(
            return_value=NS(message_id=56)), delete_message=AsyncMock(),
            set_my_commands=AsyncMock(), set_chat_menu_button=AsyncMock())
        interface = NS(owner_id=11, control_state=state,
                       application=NS(bot=api), workspace=None)
        menu = BotMenu(interface)
        await menu.show(None, "same menu", InlineKeyboardMarkup([]))
        api.edit_message_text.assert_awaited_once()
        api.send_message.assert_not_awaited()
        panel = BotPanel(interface)
        await panel.install()
        self.assertTrue(panel.hidden)
        api.delete_message.assert_awaited_once_with(chat_id=11, message_id=55)
        self.assertEqual(ControlState(11).get("panel_message_id"), 56)

    async def test_revoked_session_reuses_existing_login_menu(self):
        state = ControlState(11)
        state.set(menu_message_id=44)
        api = NS(edit_message_text=AsyncMock(), send_message=AsyncMock())
        service = AccountBot.__new__(AccountBot)
        service.application = NS(bot=api)
        await service._login_screen(11)
        api.edit_message_text.assert_awaited_once()
        api.send_message.assert_not_awaited()

    async def test_unstarted_bot_chat_does_not_stop_other_accounts(self):
        service = AccountBot.__new__(AccountBot)
        service.application = NS(bot=NS(send_message=AsyncMock(
            side_effect=BadRequest("Chat not found"))))
        await service._login_screen(11)
        self.assertIsNone(ControlState(11).get("menu_message_id"))

    async def test_saved_session_is_checked_against_owner_each_time(self):
        client = NS(connect=AsyncMock(), is_user_authorized=AsyncMock(return_value=True),
                    get_me=AsyncMock(return_value=NS(id=11)), disconnect=AsyncMock())
        service = AccountBot.__new__(AccountBot)
        service.clients = {11: client}
        self.assertEqual((await service._verify(11)).id, 11)
        client.is_user_authorized.return_value = False
        self.assertIsNone(await service._verify(11))
        client.is_user_authorized.return_value = True
        client.get_me.return_value = NS(id=22)
        with self.assertRaises(ValueError):
            await service._verify(11)
        client.disconnect.assert_awaited()

    async def test_start_routes_saved_session_without_qr_and_revoked_to_login(self):
        service = AccountBot.__new__(AccountBot)
        service.credentials = {11: object(), 22: object()}
        service.application = NS(initialize=AsyncMock(),
            bot=NS(set_my_commands=AsyncMock()),
            updater=NS(start_polling=AsyncMock()), start=AsyncMock())
        service._verify = AsyncMock(side_effect=[NS(id=11), None])
        service._ready = AsyncMock()
        service._login_screen = AsyncMock()
        await service.start()
        service._ready.assert_awaited_once()
        self.assertEqual(service._ready.call_args.args[0], 11)
        service._login_screen.assert_awaited_once_with(22)

    async def test_mismatched_owner_is_blocked_without_qr(self):
        service = AccountBot.__new__(AccountBot)
        service.credentials = {11: object(), 22: object()}
        service.blocked_owners = set()
        service.application = NS(initialize=AsyncMock(),
            bot=NS(set_my_commands=AsyncMock()),
            updater=NS(start_polling=AsyncMock()), start=AsyncMock())
        service._verify = AsyncMock(side_effect=[ValueError("wrong owner"), NS(id=22)])
        service._ready = AsyncMock()
        service._login_screen = AsyncMock()
        await service.start()
        self.assertIn(11, service.blocked_owners)
        service._login_screen.assert_not_awaited()
        service._ready.assert_awaited_once()

    async def test_first_qr_login_uses_only_matching_owner(self):
        qr = NS(url="tg://login?token=fake", wait=AsyncMock(return_value=NS(id=22)))
        client = NS(connect=AsyncMock(), qr_login=AsyncMock(return_value=qr),
                    sign_in=AsyncMock(), log_out=AsyncMock())
        image = NS(save=Mock())
        qrcode = NS(make=Mock(return_value=image))
        service = AccountBot.__new__(AccountBot)
        service.clients = {22: client}
        service.login_locks = {22: asyncio.Lock()}
        service.application = NS(bot=NS(send_photo=AsyncMock(return_value=NS(delete=AsyncMock()))))
        service._ready = AsyncMock()
        service._login_screen = AsyncMock()
        with patch.dict(sys.modules, {"qrcode": qrcode}):
            await service._qr_login(22)
        service._ready.assert_awaited_once()
        client.log_out.assert_not_awaited()
        client.sign_in.assert_not_awaited()

        qr.wait.return_value = NS(id=11)
        service._ready.reset_mock()
        with patch.dict(sys.modules, {"qrcode": qrcode}):
            await service._qr_login(22)
        service._ready.assert_not_awaited()
        client.log_out.assert_awaited_once()

    async def test_interrupted_job_pauses_without_starting_preparation(self):
        from mybot.services.runtime_controller import RuntimeController
        path = Path(self.temp.name) / "account_11" / "global" / "preparation_job.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"dialog_id": 33, "status": "running", "progress": "Embeddings 2/9"}))
        bot = NS(control_state=ControlState(11))
        controller = RuntimeController(Mock(), NS(id=11), bot)
        self.assertEqual(controller.job["status"], "paused")
        self.assertEqual(json.loads(path.read_text())["status"], "paused")
        self.assertIsNone(controller.task)

    async def test_restore_missing_dialog_preserves_saved_id(self):
        from mybot.services.runtime_controller import RuntimeController
        state = ControlState(11)
        state.set(active_dialog_id=33)
        manager = NS(find_dialog=AsyncMock(return_value=None))
        bot = NS(control_state=state, agent_manager=manager, control_bot_id=99)
        controller = RuntimeController(Mock(), NS(id=11), bot)
        with self.assertRaisesRegex(ValueError, "больше не доступен"):
            await controller.restore(33)
        self.assertEqual(ControlState(11).get("active_dialog_id"), 33)
        self.assertIsNone(controller.listener)

    async def test_restore_ready_dialog_attaches_own_listener(self):
        from mybot.services.runtime_controller import RuntimeController
        states = [ControlState(11), ControlState(22)]
        listeners = []
        for owner, state in zip((11, 22), states):
            state.set(active_dialog_id=owner * 10)
            dialog = NS(id=owner * 10, name=f"Dialog {owner}")
            bot = NS(control_state=state, agent_manager=NS(find_dialog=AsyncMock(return_value=dialog)),
                     control_bot_id=99)
            client = Mock()
            runtime = NS(dialog_name=dialog.name, workspace=NS(account_id=owner, dialog_id=dialog.id),
                         raw_history=[], recent_messages=[], known_message_ids=set(),
                         episode_tracker=object(), reply_service=object())
            with patch("mybot.services.runtime_controller.restore_ready_runtime", return_value=runtime), \
                 patch("mybot.services.runtime_controller.LiveEvents") as event_type:
                controller = RuntimeController(client, NS(id=owner), bot)
                result = await controller.restore(dialog.id)
                listeners.append(controller.listener)
                self.assertIs(result, runtime)
                self.assertEqual(bot.workspace.account_id, owner)
                event_type.return_value.register.assert_called_once()
        self.assertIsNot(listeners[0], listeners[1])


if __name__ == "__main__":
    unittest.main()

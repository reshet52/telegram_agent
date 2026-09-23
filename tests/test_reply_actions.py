"""The control bot may prepare copyable text, never send to a counterpart."""
import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch

from telegram import ReplyKeyboardRemove

from mybot.app.session_state import SessionState
from mybot.services.reply_service import ReplyService
from mybot.services.runtime_controller import RuntimeController
from mybot.storage.reply_journal import (
    get_draft, record_generation, record_outgoing_reply)
from mybot.storage.workspace import Workspace
from mybot.telegram.bot_interface import BotInterface
from mybot.telegram.bot_keyboards import control_panel, main_menu
from mybot.telegram.bot_menu import BotMenu
from mybot.telegram.bot_reply_actions import variants_markup, variants_text


def update(text='my answer', owner=1, markup=None):
    source = NS(message_id=1, text=text, reply_markup=markup, delete=AsyncMock(),
                reply_text=AsyncMock(), edit_text=AsyncMock())
    return NS(effective_message=source, effective_user=NS(id=owner),
              effective_chat=NS(id=owner, type='private'),
              callback_query=NS(answer=AsyncMock()))


class ReplyActionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.workspace = Workspace(1, 42, 'Anna', root / 'account_1/dialogs/dialog_42')
        self.other = Workspace(1, 13, 'Other', root / 'account_1/dialogs/dialog_13')
        self.bot = BotInterface(1, [], NS(), SessionState(), workspace=self.workspace,
                                telegram_client=Mock())
        self.bot.telegram_client.send_message = AsyncMock()
        self.bot.runtime_controller = NS(busy=False, lock=asyncio.Lock(),
                                         typing=NS(dialog_id=None, stop=AsyncMock()))
        self.bot.menu = NS(show=AsyncMock(), move_to_bottom=AsyncMock())
        self.bot.application = NS(bot=NS(send_message=AsyncMock(),
                                         edit_message_text=AsyncMock()))
        self.context = NS(user_data={})
        self.generation_id = record_generation(self.workspace, ['first', 'second', 'third'],
                                               None, 'gpt-6-luna')
        self.card = update(markup=variants_markup(self.generation_id))

    def table(self, name):
        with closing(sqlite3.connect(self.workspace.reply_journal)) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(f'SELECT * FROM {name}')]

    async def select(self, number=1, edit=False):
        prefix = 'ui:rve' if edit else 'ui:rv'
        await self.bot.callbacks.dispatch(f'{prefix}:{self.generation_id}:{number}',
                                          self.card,
                                          self.context)
        return self.table('reply_drafts')[-1]['draft_id']

    async def test_choose_edit_back_and_keep_only_in_control_bot(self):
        draft_id = await self.select(2)
        self.assertEqual(self.card.effective_message.edit_text.call_args.args[0],
                         'Вариант 2\nsecond')
        await self.bot.callbacks.dispatch(f'ui:draft:edit:{draft_id}', self.card, self.context)
        self.assertEqual(self.context.user_data['pending_reply_edit'], (42, draft_id, 1))
        self.assertEqual(len(self.card.effective_message.edit_text.call_args.kwargs[
            'reply_markup'].inline_keyboard), 1)
        await self.bot.callbacks.dispatch(f'ui:db:{draft_id}', self.card, self.context)
        self.assertNotIn('pending_reply_edit', self.context.user_data)
        await self.bot.callbacks.dispatch(f'ui:rve:{self.generation_id}:2', self.card, self.context)
        edited = self.table('reply_drafts')[-1]['draft_id']
        self.bot.send_generated_answers = AsyncMock()
        await self.bot.text_instruction(update(text='edited answer'), self.context)
        self.bot.send_generated_answers.assert_not_awaited()
        self.assertEqual(get_draft(self.workspace, edited)['final_text'], 'edited answer')
        self.bot.application.bot.edit_message_text.assert_awaited_once()
        await self.bot.callbacks.dispatch(f'ui:draft:keep:{edited}', self.card, self.context)
        self.assertEqual(self.card.effective_message.edit_text.call_args.args[0],
                         'Оставлен вариант 2\nedited answer')
        self.bot.application.bot.send_message.assert_not_awaited()
        self.bot.telegram_client.send_message.assert_not_awaited()
        self.assertEqual(get_draft(self.workspace, edited)['status'], 'kept')
        self.assertEqual(self.table('outgoing_replies'), [])

    async def test_options_and_cancel_return_to_variants(self):
        draft_id = await self.select()
        await self.bot.callbacks.dispatch(f'ui:draft:cancel:{draft_id}', self.card, self.context)
        self.assertEqual(get_draft(self.workspace, draft_id)['status'], 'cancelled')
        self.assertIn('Вариант 3\nthird', self.card.effective_message.edit_text.call_args.args[0])
        await self.bot.callbacks.dispatch(f'ui:opts:{self.generation_id}', self.card, self.context)
        self.assertIn('Вариант 1\nfirst', self.card.effective_message.edit_text.call_args.args[0])

    async def test_multimessage_variant_is_preserved_as_one_choice(self):
        generation_id = record_generation(self.workspace,
            ['First message\n\nSecond message\n\nThird message', 'Short', 'Another'],
            None, 'gpt-6-luna')
        card = update(markup=variants_markup(generation_id))
        await self.bot.callbacks.dispatch(f'ui:rv:{generation_id}:1', card, self.context)
        draft = self.table('reply_drafts')[-1]
        self.assertEqual(draft['original_text'],
                         'First message\n\nSecond message\n\nThird message')
        self.assertIn('Second message\n\nThird message',
                      card.effective_message.edit_text.call_args.args[0])
        await self.bot.callbacks.dispatch(f'ui:draft:edit:{draft["draft_id"]}',
                                          card, self.context)
        self.assertIn('First message\n\nSecond message\n\nThird message',
                      card.effective_message.edit_text.call_args.args[0])

    async def test_old_send_and_reconciliation_buttons_are_inert(self):
        draft_id = await self.select()
        for action in (f'ui:draft:send:{draft_id}', f'ui:draft:check:{draft_id}',
                       f'ui:dc:{draft_id}:100'):
            await self.bot.callbacks.dispatch(action, update(), self.context)
        self.bot.telegram_client.send_message.assert_not_awaited()
        self.assertEqual(get_draft(self.workspace, draft_id)['status'], 'ready')

    async def test_old_button_cannot_use_other_dialog(self):
        draft_id = await self.select()
        self.bot.workspace = self.other
        await self.bot.callbacks.dispatch(f'ui:draft:keep:{draft_id}', update(), self.context)
        self.bot.application.bot.send_message.assert_not_awaited()
        self.bot.telegram_client.send_message.assert_not_awaited()

    async def test_reject_is_reversible(self):
        await self.bot.callbacks.dispatch(f'ui:reject:{self.generation_id}', update(), self.context)
        self.assertEqual(len(self.table('generation_rejections')), 1)
        await self.bot.callbacks.dispatch(f'ui:unreject:{self.generation_id}', update(),
                                          self.context)
        self.assertEqual(len(self.table('generation_rejections')), 0)
        await self.select()
        self.assertEqual(len(self.table('reply_drafts')), 1)

    async def test_typing_toggle_preserves_pending_edit(self):
        draft_id = await self.select(edit=True)
        self.bot.runtime_controller.typing.phase = None
        self.bot.runtime_controller.typing.start = AsyncMock()
        await self.bot.callbacks.dispatch('ui:typing_toggle', update(), self.context)
        self.assertEqual(self.context.user_data['pending_reply_edit'], (42, draft_id, 1))

    async def test_outgoing_feedback_requires_explicit_link_and_current_dialog(self):
        draft_id = await self.select(1)
        await self.bot.callbacks.dispatch(f'ui:draft:keep:{draft_id}', self.card, self.context)
        record_outgoing_reply(self.workspace, {'message_id': 301, 'sender': 'Я',
            'text': 'first', 'type': 'text'})
        self.bot.callbacks.show = AsyncMock()
        await self.bot.callbacks.dispatch('ui:fb:42:301', update(), self.context)
        markup = self.bot.callbacks.show.await_args.args[2]
        link_action = markup.inline_keyboard[0][0].callback_data
        self.assertLessEqual(len(link_action.encode()), 64)
        self.assertEqual(self.table('outgoing_replies')[0]['result'], None)
        await self.bot.callbacks.dispatch(link_action, update(), self.context)
        self.assertEqual(self.table('outgoing_replies')[0]['result'], 'linked_pending')
        finalize_action = self.bot.callbacks.show.await_args.args[2].inline_keyboard[0][0].callback_data
        await self.bot.callbacks.dispatch(finalize_action, update(), self.context)
        self.assertEqual(self.table('outgoing_replies')[0]['result'],
                         'accepted_without_edit')
        self.bot.workspace = self.other
        await self.bot.callbacks.dispatch('ui:fb:42:301', update(), self.context)
        self.assertIn('Откройте исходный диалог',
                      self.bot.callbacks.show.await_args.args[1])
        record_outgoing_reply(self.other, {'message_id': 301, 'sender': 'Я',
            'text': 'different person', 'type': 'text'})
        await self.bot.callbacks.dispatch(link_action, update(), self.context)
        with closing(sqlite3.connect(self.other.reply_journal)) as connection:
            self.assertIsNone(connection.execute(
                'SELECT result FROM outgoing_replies WHERE message_id=301').fetchone()[0])

    async def test_mirrored_outgoing_has_feedback_button_but_never_sends_to_contact(self):
        await self.bot.send_outgoing_message('Anna',
            {'message_id': 501, 'sender': 'Я', 'text': 'first\n\nsecond'})
        markup = self.bot.application.bot.send_message.await_args.kwargs['reply_markup']
        self.assertEqual(markup.inline_keyboard[0][0].callback_data, 'ui:fb:42:501')
        self.bot.telegram_client.send_message.assert_not_awaited()

    async def test_generated_variant_is_kept_when_home_is_opened(self):
        menu = BotMenu(NS(owner_id=1, application=NS(bot=NS(send_message=AsyncMock()))))
        variant = update(markup=variants_markup(self.generation_id)).effective_message
        await menu.show(variant, 'home', main_menu())
        variant.delete.assert_not_awaited()
        menu.interface.application.bot.send_message.assert_awaited_once()

    async def test_generation_presents_one_message_with_numeric_choices(self):
        self.bot.reply_service = NS(
            generate=AsyncMock(return_value='sample'),
            split_variants=lambda answer: ['one', 'two', 'three'],
            last_context_message_id=None,
        )
        source = update()
        await self.bot.send_generated_answers(source)
        self.assertEqual(source.effective_message.reply_text.await_count, 1)
        call = source.effective_message.reply_text.await_args
        self.assertEqual(call.args[0], 'Вариант 1\none\n\nВариант 2\ntwo\n\nВариант 3\nthree')
        self.assertEqual([button.text for button in call.kwargs['reply_markup'].inline_keyboard[0]],
                         ['1', '2', '3'])
        self.assertEqual(variants_text(['A\n\nB', 'C', 'D']),
                         'Вариант 1\nA\n\nB\n\nВариант 2\nC\n\nВариант 3\nD')

    def test_three_variant_parsing_and_compact_keyboard(self):
        self.assertEqual(ReplyService.split_variants('A\n\nB\n\nC'), ['A', 'B', 'C'])
        self.assertEqual(ReplyService.split_variants('{"variants":["A","B","C"]}'),
                         ['A', 'B', 'C'])
        self.assertEqual(ReplyService.split_variants('A\nB\nC'), ['A', 'B', 'C'])
        self.assertEqual(ReplyService.split_variants('One answer only'), [])
        active_labels = [button.text for row in control_panel(active=True).keyboard
                         for button in row]
        self.assertNotIn('💬 Диалоги', active_labels)
        self.assertIn('🫥 Скрыть пульт', active_labels)
        global_labels = [button.text for row in control_panel(active=False).keyboard
                         for button in row]
        self.assertIn('💬 Диалоги', global_labels)
        self.assertNotIn('✍️ Ответ', global_labels)

    async def test_hide_and_show_panel(self):
        await self.bot.panel.hide()
        self.assertTrue(self.bot.panel.hidden)
        self.assertIsInstance(
            self.bot.application.bot.send_message.await_args.kwargs['reply_markup'],
            ReplyKeyboardRemove)
        await self.bot.panel.show()
        self.assertFalse(self.bot.panel.hidden)
        labels = [button.text for row in
                  self.bot.application.bot.send_message.await_args.kwargs['reply_markup'].keyboard
                  for button in row]
        self.assertNotIn('💬 Диалоги', labels)

    async def test_leave_deactivates_runtime_and_returns_global_menu(self):
        with patch('mybot.services.runtime_controller.get_global_root',
                   return_value=Path(self.temporary.name)):
            controller = RuntimeController(Mock(), NS(id=1), self.bot)
        self.bot.runtime_controller = controller
        listener = NS(unregister=AsyncMock())
        controller.listener = listener
        controller.runtime = object()
        self.bot.reply_service = object()
        self.bot.agent_manager.selected_dialog = NS(id=42)
        await self.bot.callbacks.dispatch('ui:leave', update(), self.context)
        listener.unregister.assert_awaited_once()
        self.assertIsNone(self.bot.workspace)
        self.assertIsNone(self.bot.reply_service)
        self.assertIsNone(self.bot.agent_manager.selected_dialog)
        self.assertTrue(any(button.callback_data == 'ui:dialogs:0'
                            for row in self.bot.menu.show.await_args.args[2].inline_keyboard
                            for button in row))

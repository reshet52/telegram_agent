"""Prepare one isolated runtime; the control bot owns all user interaction."""

import asyncio
import logging
from dataclasses import dataclass
from mybot.episodes.incremental import (initialize_episode_tracker, IncrementalEpisodeTracker,
    get_last_episode_info, find_history_start, PREVIOUS_CONTEXT_LIMIT)
from mybot.episodes.maintenance import ensure_episode_embeddings
from mybot.memory.manager import load_agent_memory
from mybot.services.reply_service import ReplyService
from mybot.services.history_import import import_history
from mybot.services.workspace_initializer import WorkspaceInitializer, is_prepared, has_legacy_memory
from mybot.storage.atomic import read_json
from mybot.storage.deletions import check_recent_deletions, mark_messages_deleted
from mybot.storage.history import load_all_messages, load_last_messages
from mybot.storage.reply_journal import record_outgoing_replies
from mybot.storage.workspace import create_workspace, get_workspace_root
from mybot.services.workspace_initializer import has_legacy_memory
from mybot.storage.workspace import Workspace


@dataclass
class DialogRuntime:
    dialog_name: str
    workspace: object
    raw_history: list
    recent_messages: list
    known_message_ids: set
    reply_service: ReplyService
    bot_interface: object
    episode_tracker: object


async def prepare_dialog_runtime(client, selected_dialog, me, state,
                                 recent_messages_limit=15, bot_interface=None,
                                 progress=None, full_analysis=False):
    async def report(text):
        if progress:
            await progress(text)
    name = selected_dialog.name or f'Dialog {selected_dialog.id}'
    workspace = create_workspace(me.id, selected_dialog.id, name)
    await report('История: загружаю сообщения…')
    await import_history(client, selected_dialog, me.id, workspace, report)
    raw_history = await asyncio.to_thread(load_all_messages, filename=workspace.chat_history, include_deleted=True,
                                    deleted_filename=workspace.deleted_message_ids)
    try:
        await asyncio.to_thread(record_outgoing_replies, workspace, raw_history)
    except Exception:
        logging.exception("Не удалось восстановить исходящие сообщения в журнале диалога")
    await report('Проверка удалённых сообщений…')
    deleted = await check_recent_deletions(client=client, dialog_id=selected_dialog.id,
        messages=raw_history, deleted_filename=workspace.deleted_message_ids)
    mark_messages_deleted(deleted, filename=workspace.deleted_message_ids)
    await report('Подготавливаю эпизоды и embeddings…')
    tracker, _ = await initialize_episode_tracker(
        history_filename=workspace.chat_history, episodes_filename=workspace.episodes,
        live_embeddings_filename=workspace.episode_embeddings_live,
        deleted_filename=workspace.deleted_message_ids, progress=report)
    await ensure_episode_embeddings(workspace, report)
    initial_state = read_json(workspace.root / 'initialization_state.json', {})
    needs_analysis = (not is_prepared(workspace) and
                      (bool(initial_state) or not has_legacy_memory(workspace)))
    if full_analysis or needs_analysis:
        await WorkspaceInitializer(workspace, report).run()
    recent = load_last_messages(filename=workspace.chat_history, count=recent_messages_limit,
                               deleted_filename=workspace.deleted_message_ids)
    memory = load_agent_memory(user_profile_filename=workspace.user_profile,
                               person_profile_filename=workspace.person_profile)
    return DialogRuntime(
        dialog_name=name, workspace=workspace, raw_history=raw_history, recent_messages=recent,
        known_message_ids={m['message_id'] for m in raw_history if m.get('message_id') is not None},
        reply_service=ReplyService(recent, memory, workspace=workspace),
        bot_interface=bot_interface, episode_tracker=tracker)


def restore_ready_runtime(selected_dialog, me, bot_interface=None):
    """Load a completed workspace without imports, indexing or OpenAI calls."""
    root = get_workspace_root(me.id, selected_dialog.id)
    name = selected_dialog.name or f"Dialog {selected_dialog.id}"
    workspace = Workspace(me.id, selected_dialog.id, name, root)
    if not workspace.chat_history.exists() or not (is_prepared(workspace) or has_legacy_memory(workspace)):
        raise ValueError("Подготовка диалога не завершена. Нажмите «Продолжить» в статусе.")
    raw_history = load_all_messages(filename=workspace.chat_history, include_deleted=True,
                                    deleted_filename=workspace.deleted_message_ids)
    active_history = load_all_messages(filename=workspace.chat_history,
                                       deleted_filename=workspace.deleted_message_ids)
    last_episode_id, last_message_id = get_last_episode_info(workspace.episodes)
    # Missing nonzero checkpoint ID must remain a hard error.
    start = find_history_start(active_history, last_message_id)
    tracker = IncrementalEpisodeTracker(last_episode_id + 1,
        active_history[max(0, start - PREVIOUS_CONTEXT_LIMIT):start])
    # Reconstruct tracker state in memory; don't persist episodes or create embeddings.
    for message in active_history[start:]:
        tracker.process_message(message)
    recent = load_last_messages(filename=workspace.chat_history, count=15,
                                deleted_filename=workspace.deleted_message_ids)
    memory = load_agent_memory(user_profile_filename=workspace.user_profile,
                               person_profile_filename=workspace.person_profile)
    return DialogRuntime(name, workspace, raw_history, recent,
        {m['message_id'] for m in raw_history if m.get('message_id') is not None},
        ReplyService(recent, memory, workspace=workspace), bot_interface, tracker)

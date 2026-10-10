"""Incremental history/episode update; no full analysis or memory replacement."""

import asyncio

from mybot.episodes.incremental import initialize_episode_tracker, get_last_episode_info
from mybot.services.history_import import commit_batch
from mybot.storage.history import get_last_saved_message_id, load_all_messages
from mybot.storage.atomic import durable_call, read_json, write_text
from mybot.services.message_order import ordered_messages
import json
from mybot.storage.history_transaction import HistoryTransaction
from mybot.telegram.exporter import fetch_messages_after


async def stage_history_update(client, me_id, dialog, workspace, progress):
    transaction = HistoryTransaction(workspace)
    staged = transaction.staged
    try:
        from mybot.episodes.checkpoint import recover_batch
        recover_batch(staged.episodes, staged.episode_embeddings_live)
        last_id = get_last_saved_message_id(staged.chat_history)
        if not last_id:
            raise ValueError('Нет сохранённой истории. Полная загрузка через обновление запрещена.')
        await progress('Обновление новой истории: проверяю новые сообщения…')
        idle = read_json(workspace.root / 'learning' / 'idle_history.json', {})
        recovery = idle.get('status') == 'recovery_required'
        cursor = idle.get('verified_id') if recovery else last_id
        if type(cursor) is not int or cursor <= 0 or cursor > last_id:
            raise ValueError('Некорректная контрольная точка дозагрузки истории.')
        messages = await fetch_messages_after(client, dialog, cursor, me_id)
        existing = load_all_messages(staged.chat_history, include_deleted=True)
        known = {item['message_id'] for item in existing}
        missing = []
        for item in messages:
            if item['message_id'] not in known:
                missing.append(item)
                known.add(item['message_id'])
        if missing and recovery:
            ordered = ordered_messages(existing + missing)
            write_text(staged.chat_history, ''.join(json.dumps(m, ensure_ascii=False, indent=4) + '\n\n'
                                                   for m in ordered))
        elif missing:
            await durable_call(commit_batch, staged.chat_history, missing)
        _, checkpoint = get_last_episode_info(staged.episodes)
        if not checkpoint and len(existing) > 200:
            raise ValueError('Нет контрольной точки эпизодов. Автоматическая пересборка всей истории запрещена.')
        await progress(f'Новых сообщений: {len(missing)}. Проверяю незаписанные эпизоды…')
        if recovery and missing and min(m['message_id'] for m in missing) <= checkpoint:
            from mybot.services.history_recovery import repair_episodes
            tracker, episodes = await repair_episodes(transaction, ordered,
                min(m['message_id'] for m in missing), progress)
        else:
            tracker, episodes = await initialize_episode_tracker(
                history_filename=staged.chat_history, episodes_filename=staged.episodes,
                live_embeddings_filename=staged.episode_embeddings_live,
                deleted_filename=staged.deleted_message_ids, progress=progress)
        await progress(f'Готово к сохранению: сообщений {len(missing)}, эпизодов {len(episodes)}.')
        await asyncio.sleep(0)  # Last cancellation point before synchronous publication.
        return transaction, tracker, missing, len(episodes)
    except BaseException:
        transaction.close()
        raise

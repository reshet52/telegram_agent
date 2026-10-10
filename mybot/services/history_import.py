"""Resumable history import, compatible with the existing blank-line JSON format."""

import json
from mybot.storage.atomic import write_text, durable_call
from mybot.storage.history import get_last_saved_message_id
from mybot.telegram.exporter import message_to_data


class HistoryRecoveryRequired(ValueError):
    """A gap precedes episodes already derived from the saved history."""


def commit_batch(path, messages):
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    # Preserve legacy bytes while atomically adding a complete batch.
    text = previous.rstrip() + '\n\n' if previous.strip() else ''
    text += ''.join(json.dumps(item, ensure_ascii=False, indent=4) + '\n\n'
                    for item in messages)
    write_text(path, text)


def merge_out_of_order_batch(workspace, messages):
    """Repair a missed live interval transactionally, only beyond the episode checkpoint."""
    from mybot.storage.history import load_all_messages
    from mybot.storage.history_transaction import HistoryTransaction
    from mybot.episodes.incremental import get_last_episode_info
    _, checkpoint = get_last_episode_info(workspace.episodes)
    if checkpoint and min(m['message_id'] for m in messages) <= checkpoint:
        raise HistoryRecoveryRequired('Пропущенные сообщения старше контрольной точки эпизодов. '
                                      'Нужно явное восстановление, автоматическая пересборка запрещена.')
    transaction = HistoryTransaction(workspace)
    try:
        existing = load_all_messages(transaction.staged.chat_history, include_deleted=True)
        ids = {m['message_id'] for m in existing}
        if len(ids) != len(existing) or ids.intersection(m['message_id'] for m in messages):
            raise ValueError('История содержит повторяющиеся ID; автодозагрузка остановлена.')
        ordered = sorted(existing + list(messages), key=lambda m: m['message_id'])
        content = ''.join(json.dumps(m, ensure_ascii=False, indent=4) + '\n\n' for m in ordered)
        write_text(transaction.staged.chat_history, content)
        transaction.publish()
    finally:
        transaction.close()


async def import_history(client, dialog, me_id, workspace, progress):
    last_id = get_last_saved_message_id(workspace.chat_history)
    batch, added = [], 0
    async for message in client.iter_messages(dialog.id, min_id=last_id, reverse=True, limit=None):
        batch.append(await message_to_data(message, me_id, dialog.name))
        if len(batch) >= 200:
            await durable_call(commit_batch, workspace.chat_history, batch)
            added += len(batch)
            batch = []
            await progress(f'История: сохранено ещё {added} сообщений…')
    if batch:
        await durable_call(commit_batch, workspace.chat_history, batch)
        added += len(batch)
    if not workspace.chat_history.exists():
        write_text(workspace.chat_history, '')
    await progress(f'История ✅ Добавлено сообщений: {added}')

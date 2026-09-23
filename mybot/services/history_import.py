"""Resumable history import, compatible with the existing blank-line JSON format."""

import json
from mybot.storage.atomic import write_text, durable_call
from mybot.storage.history import get_last_saved_message_id
from mybot.telegram.exporter import message_to_data


def commit_batch(path, messages):
    previous = path.read_text(encoding='utf-8') if path.exists() else ''
    # Preserve legacy bytes while atomically adding a complete batch.
    text = previous.rstrip() + '\n\n' if previous.strip() else ''
    text += ''.join(json.dumps(item, ensure_ascii=False, indent=4) + '\n\n'
                    for item in messages)
    write_text(path, text)


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

"""Owner-written reply direction, isolated from factual memory and other dialogs."""

from datetime import datetime, timezone

from mybot.storage.atomic import read_json, write_json


MAX_GOAL_LENGTH = 2500


def load_goal(workspace):
    if workspace is None:
        return ''
    value = read_json(workspace.root / 'dialog_goal.json', None)
    if value is None:
        return ''
    if (not isinstance(value, dict) or value.get('version') != 1
            or value.get('account_id') != workspace.account_id
            or value.get('dialog_id') != workspace.dialog_id
            or not isinstance(value.get('text'), str)
            or len(value['text']) > MAX_GOAL_LENGTH):
        raise ValueError('Файл цели повреждён или относится к другому диалогу. Генерация с этой целью недоступна.')
    return value['text']


def save_goal(workspace, text):
    if not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_GOAL_LENGTH:
        raise ValueError(f'Напишите цель от 1 до {MAX_GOAL_LENGTH} символов.')
    _write(workspace, text.strip())


def clear_goal(workspace):
    _write(workspace, '')


def _write(workspace, text):
    # Do not overwrite a corrupt or accidentally copied file silently.
    load_goal(workspace)
    write_json(workspace.root / 'dialog_goal.json', {
        'version': 1, 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
        'text': text, 'updated_at': datetime.now(timezone.utc).isoformat(),
    })

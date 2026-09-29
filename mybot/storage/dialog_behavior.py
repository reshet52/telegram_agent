"""Owner-defined behavior and current state; never inferred into factual memory."""

from datetime import datetime, timezone

from mybot.storage.atomic import read_json, write_json


DEFAULTS = {
    'directness': 'natural', 'persistence': 'moderate',
    'boundaries': '', 'stop_condition': '', 'current_state': '',
}
CHOICES = {
    'directness': {'natural': 'По моему стилю', 'direct': 'Прямо'},
    'persistence': {'moderate': 'Умеренно', 'persistent': 'Настойчиво'},
}
LABELS = {
    'directness': 'Прямота', 'persistence': 'Настойчивость',
    'boundaries': 'Ограничения и полномочия', 'stop_condition': 'Когда остановиться',
    'current_state': 'Моё текущее состояние',
}
MAX_TEXT_LENGTH = 800


def validate(values):
    if not isinstance(values, dict) or set(values) != set(DEFAULTS):
        raise ValueError('Некорректные настройки поведения.')
    for field, value in values.items():
        if not isinstance(value, str):
            raise ValueError('Некорректные настройки поведения.')
        if field in CHOICES:
            if value not in CHOICES[field]:
                raise ValueError('Неизвестный режим поведения.')
        elif len(value) > MAX_TEXT_LENGTH:
            raise ValueError(f'Допустимо до {MAX_TEXT_LENGTH} символов.')
    return dict(values)


def load_behavior(workspace):
    if workspace is None:
        return dict(DEFAULTS)
    value = read_json(workspace.root / 'dialog_behavior.json')
    if value is None:
        return dict(DEFAULTS)
    if (not isinstance(value, dict) or value.get('version') != 1
            or value.get('account_id') != workspace.account_id
            or value.get('dialog_id') != workspace.dialog_id):
        raise ValueError('Настройки поведения повреждены или относятся к другому диалогу.')
    return validate(value.get('values'))


def save_behavior_field(workspace, field, value):
    if field not in DEFAULTS or not isinstance(value, str):
        raise ValueError('Неизвестная настройка поведения.')
    values = load_behavior(workspace)
    values[field] = value.strip()
    validate(values)
    write_json(workspace.root / 'dialog_behavior.json', {
        'version': 1, 'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
        'values': values, 'updated_at': datetime.now(timezone.utc).isoformat(),
    })


def format_behavior(values):
    return '\n'.join(f'{LABELS[key]}: {CHOICES[key][value] if key in CHOICES else value or "Не задано"}'
                     for key, value in values.items())

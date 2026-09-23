"""Account-wide style: only numeric/closed-vocabulary traits, never dialog facts."""

from collections import Counter
from mybot.storage.atomic import read_json, write_json
from mybot.storage.workspace import get_global_root


TRAITS = {
    'length': {'short', 'medium', 'long'},
    'formality': {'informal', 'neutral', 'formal'},
    'emoji': {'rare', 'moderate', 'frequent'},
    'punctuation': {'minimal', 'standard', 'expressive'},
    'message_splitting': {'single', 'multiple'},
}


def validate_traits(value):
    if not isinstance(value, dict) or set(value) != set(TRAITS):
        raise ValueError('Некорректная структура общего стиля.')
    if any(not isinstance(value[key], str) or value[key] not in allowed
           for key, allowed in TRAITS.items()):
        raise ValueError('Общий стиль содержит недопустимые значения.')
    return {key: value[key] for key in TRAITS}


def publish_style(workspace, observations, message_count):
    # Each contribution is replaceable; reopening a dialog never increases its weight.
    root = get_global_root(workspace.account_id)
    contributions = read_json(root / 'style_sources.json', {})
    counts = {key: dict(Counter(item[key] for item in observations)) for key in TRAITS}
    contributions[str(workspace.dialog_id)] = {'counts': counts, 'messages': message_count}
    write_json(root / 'style_sources.json', contributions)
    totals = {key: Counter() for key in TRAITS}
    for source in contributions.values():
        for key, allowed in TRAITS.items():
            for value, count in source['counts'].get(key, {}).items():
                if value in allowed and isinstance(count, int) and count > 0:
                    totals[key][value] += count
    traits = {key: values.most_common(1)[0][0] for key, values in totals.items() if values}
    write_json(root / 'user_style.json', {
        'version': 1, 'traits': traits,
        'analyzed_messages': sum(item['messages'] for item in contributions.values()),
        'dialog_count': len(contributions),
    })


def load_global_style(account_id):
    profile = read_json(get_global_root(account_id) / 'user_style.json', {})
    traits = profile.get('traits', {})
    # Only the allowlist enters a prompt, even if a file was edited externally.
    return {key: value for key, value in traits.items()
            if key in TRAITS and isinstance(value, str) and value in TRAITS[key]}

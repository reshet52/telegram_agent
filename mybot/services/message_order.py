"""Canonical ordering for messages in one Telegram dialog."""

from copy import deepcopy


def ordered_messages(messages):
    # IDs are chronological within a dialog. Preserve ID-less diagnostic rows.
    by_id = {}
    without_id = []
    for message in messages:
        key = message.get('message_id')
        if isinstance(key, int):
            by_id[key] = message
        else:
            without_id.append(message)
    return [by_id[key] for key in sorted(by_id)] + without_id


def context_snapshot(messages):
    return deepcopy(ordered_messages(messages))

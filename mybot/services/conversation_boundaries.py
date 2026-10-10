"""Time boundaries for historical reply evidence, scoped to one dialog."""

from datetime import datetime

from mybot.storage.atomic import read_json, write_json

DEFAULT_GAP_SECONDS = 4 * 60 * 60


def gap_seconds(workspace=None):
    if workspace is None:
        return DEFAULT_GAP_SECONDS
    data = read_json(workspace.root / 'learning' / 'session_policy.json') or {}
    value = data.get('gap_seconds', DEFAULT_GAP_SECONDS)
    if type(value) is not int or not 15 * 60 <= value <= 24 * 60 * 60:
        raise ValueError('Порог паузы должен быть от 15 минут до 24 часов.')
    return value


def set_gap_seconds(workspace, value):
    if type(value) is not int or not 15 * 60 <= value <= 24 * 60 * 60:
        raise ValueError('Порог паузы должен быть от 15 минут до 24 часов.')
    write_json(workspace.root / 'learning' / 'session_policy.json',
               {'account_id': workspace.account_id, 'dialog_id': workspace.dialog_id,
                'gap_seconds': value})


def parse_date(message):
    value = message.get('date')
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def elapsed(before, after):
    first, second = parse_date(before), parse_date(after)
    if not first or not second or not first.tzinfo or not second.tzinfo:
        return None
    return (second - first).total_seconds()


def boundary(before, after, threshold=DEFAULT_GAP_SECONDS):
    delta = elapsed(before, after)
    return delta is not None and (delta < 0 or delta >= threshold)


def sessions(messages, threshold=DEFAULT_GAP_SECONDS):
    """Return contiguous chronological slices where timestamps establish a gap."""
    if not messages:
        return []
    starts = [0]
    for index in range(1, len(messages)):
        if boundary(messages[index - 1], messages[index], threshold):
            starts.append(index)
    return [messages[start:end] for start, end in zip(starts, starts[1:] + [len(messages)])]


def crossing_reason(context, expected, threshold=DEFAULT_GAP_SECONDS):
    if not context or not expected:
        return 'missing_reply'
    if boundary(context[-1], expected[0], threshold):
        return 'reply_after_long_pause'
    if any(boundary(a, b, threshold) for a, b in zip(context + expected, (context + expected)[1:])):
        return 'context_crosses_session'
    return None


def wait_review_candidates(messages, deleted, *, threshold=DEFAULT_GAP_SECONDS, max_context=24):
    """Unanswered incoming session endings with an observed following session.

    They are questions for the owner, never automatic WAIT labels.
    """
    result = []
    parts = sessions(messages, threshold)
    for segment, following in zip(parts, parts[1:]):
        if not segment or segment[-1].get('sender') == 'Я':
            continue
        delta = elapsed(segment[-1], following[0])
        if delta is None or delta < threshold or delta < 0:
            continue
        context = segment[-max_context:]
        if (any(m['message_id'] in deleted or m.get('type') not in (None, 'text')
                or not isinstance(m.get('text'), str) or not m['text'].strip() for m in context)
                or sum(len(m['text']) for m in context) > 14000):
            continue
        result.append((context, {'reason': 'incoming_before_observed_long_pause',
                                 'next_message_id': following[0]['message_id'],
                                 'gap_seconds': int(delta)}))
    return result

"""Observed timing, never an automatic label of mood or intent."""

from datetime import datetime, timezone
from statistics import median

from mybot.services.conversation_boundaries import parse_date, elapsed, sessions, gap_seconds
from mybot.storage.atomic import read_json, write_json


def context_timing(messages, *, reply_time=None, threshold=14400):
    turns = []
    for i, message in enumerate(messages):
        date = parse_date(message)
        turns.append({'source_id': message['message_id'],
                      'sent_at': date.isoformat() if date and date.tzinfo else None,
                      'gap_seconds': elapsed(messages[i - 1], message) if i else None})
    segment = sessions(messages, threshold)[-1] if messages else []
    age = elapsed(messages[-1], {'date': reply_time}) if messages and reply_time else None
    return {'messages': turns, 'session_gap_threshold_seconds': threshold,
            'observed_session_start': segment[0].get('date') if segment else None,
            'observed_duration_seconds': elapsed(segment[0], segment[-1]) if segment else None,
            'seconds_since_last_message': age if age is not None and age >= 0 else None,
            'interpretation': 'Пауза не доказывает завершение разговора или настроение. Начало сессии может быть вне показанного контекста.'}


def history_report(ws):
    from mybot.storage.reply_practice import PracticeStore, digest
    from mybot.storage.deletions import load_deleted_message_ids
    messages = PracticeStore(ws).history()
    deleted = load_deleted_message_ids(ws.deleted_message_ids)
    fingerprint = digest([messages, sorted(deleted), gap_seconds(ws)])
    path = ws.root / 'reply_assistant/timing_report.json'
    old = read_json(path, {})
    if old and (old.get('account_id'), old.get('dialog_id')) != (ws.account_id, ws.dialog_id):
        raise ValueError('Чужой отчёт времени в каталоге диалога.')
    if old.get('source_digest') == fingerprint:
        return old
    eligible = [m for m in messages if m['message_id'] not in deleted and parse_date(m) and parse_date(m).tzinfo]
    parts = sessions(eligible, gap_seconds(ws))
    eligible_ids = {m['message_id'] for m in eligible}
    records, latencies = [], []
    for part in parts:
        records.append({'first_id': part[0]['message_id'], 'last_id': part[-1]['message_id'],
                        'start': part[0]['date'], 'end': part[-1]['date'], 'messages': len(part),
                        'duration_seconds': elapsed(part[0], part[-1]),
                        'ending_sender': part[-1]['sender']})
        for before, after in zip(part, part[1:]):
            if before['sender'] != 'Я' and after['sender'] == 'Я':
                delta = elapsed(before, after)
                if delta is not None and delta >= 0:
                    latencies.append(delta)
    result = {'account_id': ws.account_id, 'dialog_id': ws.dialog_id, 'source_digest': fingerprint,
              'created_at': datetime.now(timezone.utc).isoformat(), 'gap_threshold_seconds': gap_seconds(ws),
              'sources': len(messages), 'dated_sources': len(eligible),
              'excluded': [{'source_id': m['message_id'], 'reason': 'deleted' if m['message_id'] in deleted else 'missing_or_unzoned_date'}
                           for m in messages if m['message_id'] not in eligible_ids],
              'sessions': records, 'owner_reply_delays_count': len(latencies),
              'median_owner_reply_delay_seconds': median(latencies) if latencies else None,
              'note': 'Сессии разделены наблюдаемыми паузами, это не подтверждённые границы разговора. Настроение по времени не определяется.'}
    write_json(path, result)
    return result


def display_context(messages):
    lines = []
    previous = None
    for message in messages:
        date = parse_date(message)
        delta = elapsed(previous, message) if previous else None
        if delta is not None and delta >= 300:
            lines.append(f'Пауза: {round(delta / 60)} мин.')
        stamp = date.strftime('%Y-%m-%d %H:%M %z') if date and date.tzinfo else 'время неизвестно'
        lines.append(f"[{stamp}] {message['sender']}: {message['text']}")
        previous = message
    return '\n\n'.join(lines)

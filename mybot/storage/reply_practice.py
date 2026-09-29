"""Historical reply exercises, isolated from live reply feedback and memory."""

import hashlib
import json
import random
import re
from datetime import datetime, timezone
from uuid import uuid4

from mybot.storage.atomic import read_json, write_json
from mybot.storage.history import load_all_messages
from mybot.storage.deletions import load_deleted_message_ids


RATINGS = {'good': 'Отправил бы', 'style': 'Не мой стиль',
           'reaction': 'Не та реакция', 'facts': 'Выдуманы факты'}
MAX_CONTEXT_MESSAGES = 24
MAX_CONTEXT_CHARS = 14000
SOURCE_FIELDS = ('message_id', 'sender', 'text', 'date', 'type')


def now():
    return datetime.now(timezone.utc).isoformat()


def source_message(message):
    value = {field: message.get(field) for field in SOURCE_FIELDS}
    value['type'] = value['type'] or 'text'
    return value


def digest(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def candidates(messages, deleted, used):
    """Hold out a complete outgoing turn; never join across a deleted message."""
    groups = []
    for index, message in enumerate(messages):
        outgoing = message.get('sender') == 'Я'
        if not groups or groups[-1][0] != outgoing:
            groups.append((outgoing, index, index + 1))
        else:
            groups[-1] = (outgoing, groups[-1][1], index + 1)
    result = []
    for index, (outgoing, start, end) in enumerate(groups):
        if not outgoing or index == 0 or messages[start]['message_id'] in used:
            continue
        # Include the complete incoming turn plus preceding context when it fits.
        context_start = max(0, start - MAX_CONTEXT_MESSAGES)
        if groups[index - 1][1] < context_start:
            continue
        context, expected = messages[context_start:start], messages[start:end]
        combined = context + expected
        if (not context or len(expected) > 10
                or any(m['message_id'] in deleted for m in combined)
                or any(m.get('type') not in {None, 'text'} or not isinstance(m.get('text'), str)
                       or not m['text'].strip() for m in combined)
                or sum(len(m['text']) for m in context) > MAX_CONTEXT_CHARS
                or sum(len(m['text']) for m in expected) > 6000):
            continue
        result.append((context, expected))
    return result


class PracticeStore:
    def __init__(self, workspace):
        self.workspace = workspace
        self.root = workspace.root / 'reply_practice'

    def path(self, key):
        if not isinstance(key, str) or not re.fullmatch(r'[a-f0-9]{32}', key):
            raise ValueError('Отрывок недоступен. Откройте тренировку заново.')
        return self.root / f'{key}.json'

    def get(self, key):
        value = read_json(self.path(key))
        if (not isinstance(value, dict) or value.get('version') != 1 or value.get('id') != key
                or value.get('account_id') != self.workspace.account_id
                or value.get('dialog_id') != self.workspace.dialog_id):
            raise ValueError('Отрывок отсутствует, повреждён или относится к другому диалогу.')
        for field in ('context_ids', 'expected_ids'):
            ids = value.get(field)
            if (not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids)
                    or ids != sorted(set(ids))):
                raise ValueError('Повреждены ссылки на сообщения отрывка.')
        if (max(value['context_ids']) >= min(value['expected_ids'])
                or not isinstance(value.get('source_digest'), str)
                or not isinstance(value.get('ratings'), dict)
                or not isinstance(value.get('variants'), list)):
            raise ValueError('Повреждён сохранённый отрывок.')
        return value

    def records(self):
        return sorted((self.get(path.stem) for path in self.root.glob('*.json')),
                      key=lambda case: case['created_at'], reverse=True)

    def history(self):
        messages = load_all_messages(filename=self.workspace.chat_history, include_deleted=True)
        ids = [message.get('message_id') for message in messages]
        if any(type(key) is not int for key in ids) or len(set(ids)) != len(ids):
            raise ValueError('История содержит отсутствующие или повторяющиеся ID сообщений.')
        return [source_message(m) for m in sorted(messages, key=lambda m: m['message_id'])]

    def create(self):
        used = {case['expected_ids'][0] for case in self.records()}
        options = candidates(self.history(), load_deleted_message_ids(self.workspace.deleted_message_ids), used)
        if not options:
            raise ValueError('Новых подходящих текстовых отрывков нет. Можно вернуться к сохранённым.')
        context, expected = random.choice(options)
        case = {'version': 1, 'id': uuid4().hex, 'account_id': self.workspace.account_id,
                'dialog_id': self.workspace.dialog_id, 'created_at': now(),
                'context_ids': [m['message_id'] for m in context],
                'expected_ids': [m['message_id'] for m in expected],
                'source_digest': digest(context + expected), 'status': 'prepared',
                'variants': [], 'ratings': {}, 'revealed_at': None}
        write_json(self.path(case['id']), case)
        return case

    def source(self, case):
        messages = {m['message_id']: m for m in self.history()}
        deleted = load_deleted_message_ids(self.workspace.deleted_message_ids)
        ids = case['context_ids'] + case['expected_ids']
        if any(key not in messages or key in deleted for key in ids):
            raise ValueError('Источник удалён или отсутствует. Этот отрывок больше недоступен.')
        combined = [messages[key] for key in ids]
        if digest(combined) != case['source_digest']:
            raise ValueError('Исходная переписка изменилась. Этот отрывок больше недоступен.')
        return combined[:len(case['context_ids'])], combined[len(case['context_ids']):]

    def save_request(self, key, request):
        case = self.get(key)
        self.source(case)
        if case['variants']:
            raise ValueError('Ответ уже сохранён; повторный запрос не требуется.')
        case.update(request=request, status='prepared')
        write_json(self.path(key), case)

    def save_result(self, key, variants):
        case = self.get(key)
        self.source(case)
        if case['variants']:
            return case
        if (len(variants) != 3 or any(not isinstance(v, str) or not v.strip() for v in variants)
                or len(set(variants)) != 3):
            raise ValueError('Модель не вернула три разных варианта.')
        case.update(variants=variants, status='generated', generated_at=now())
        write_json(self.path(key), case)
        return case

    def rate(self, key, number, rating):
        case = self.get(key)
        self.source(case)
        if number not in {1, 2, 3} or len(case['variants']) != 3 or rating not in RATINGS:
            raise ValueError('Оценка или вариант недоступны.')
        case['ratings'][str(number)] = {'rating': rating, 'at': now(),
                                       'after_reveal': bool(case['revealed_at'])}
        write_json(self.path(key), case)
        return case

    def reveal(self, key):
        case = self.get(key)
        self.source(case)
        if not case['variants']:
            raise ValueError('Сначала получите варианты модели, затем откройте свой ответ.')
        if not case['revealed_at']:
            case['revealed_at'] = now()
            write_json(self.path(key), case)
        return case

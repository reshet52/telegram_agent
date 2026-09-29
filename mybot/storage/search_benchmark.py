"""Private, atomic benchmark records. Never modifies production history/indexes."""

from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

from mybot.storage.atomic import read_json, write_json
from mybot.storage.history import load_all_messages


def now():
    return datetime.now(timezone.utc).isoformat()


class BenchmarkStore:
    def __init__(self, workspace):
        self.workspace = workspace
        self.path = workspace.root / 'search_benchmark.json'
        self.runs_path = workspace.root / 'search_benchmark_runs.json'

    def _read(self, path):
        value = read_json(path, None)
        if value is None:
            return {'version': 1, 'account_id': self.workspace.account_id,
                    'dialog_id': self.workspace.dialog_id, 'facts': {}, 'cases': {}, 'runs': {}}
        if (not isinstance(value, dict) or value.get('version') != 1
                or value.get('account_id') != self.workspace.account_id
                or value.get('dialog_id') != self.workspace.dialog_id):
            raise ValueError('Файл проверки относится к другому аккаунту/диалогу или повреждён.')
        for field in ('facts', 'cases', 'runs'):
            if not isinstance(value.get(field), dict):
                raise ValueError('Повреждён файл проверки поиска.')
        return value

    def save_fact(self, fact):
        data = self._read(self.path)
        key = uuid4().hex[:16]
        data['facts'][key] = {'id': key, 'expected_answer': fact.text,
                              'sources': [{field: source.get(field) for field in
                                           ('message_id', 'sender', 'date', 'quote')}
                                          for source in fact.sources], 'created_at': now()}
        write_json(self.path, data)
        return key

    def fact(self, key):
        return self._get(self._read(self.path)['facts'], key)

    @staticmethod
    def _get(items, key):
        if key not in items:
            raise ValueError('Запись недоступна. Откройте раздел проверки заново.')
        return deepcopy(items[key])

    def cases(self):
        return list(self._read(self.path)['cases'].values())

    def case(self, key):
        return self._get(self._read(self.path)['cases'], key)

    def validate_sources(self, case):
        messages = {item['message_id']: item for item in load_all_messages(
            filename=self.workspace.chat_history, deleted_filename=self.workspace.deleted_message_ids)}
        sources = case.get('sources', [])
        if not sources:
            raise ValueError('У записи нет источников.')
        for source in sources:
            original = messages.get(source.get('message_id'))
            quote = source.get('quote')
            if original is None or not isinstance(quote, str) or not quote.strip() or quote not in (original.get('text') or ''):
                raise ValueError('Источник удалён, изменён или отсутствует в истории этого диалога. Запись не сохранена/не запущена.')

    def save_case(self, draft):
        question = draft.get('question', '').strip()
        expected = draft.get('expected_answer', '').strip()
        if not 1 <= len(question) <= 1000 or not 1 <= len(expected) <= 1000:
            raise ValueError('Вопрос и ожидаемый ответ должны содержать от 1 до 1000 символов.')
        self.validate_sources(draft)
        data = self._read(self.path)
        key = draft.get('case_id') or uuid4().hex[:16]
        if draft.get('case_id') and key not in data['cases']:
            raise ValueError('Запись уже удалена.')
        existing = data['cases'].get(key, {})
        if existing and existing.get('revision') != draft.get('revision'):
            raise ValueError('Запись уже изменена. Откройте её заново.')
        record = {'id': key, 'question': question, 'expected_answer': expected,
                  'expected_source_message_ids': [s['message_id'] for s in draft['sources']],
                  'sources': deepcopy(draft['sources']), 'created_at': existing.get('created_at', now()),
                  'updated_at': now(), 'revision': existing.get('revision', 0) + 1}
        data['cases'][key] = record
        write_json(self.path, data)
        return record

    def delete_case(self, key):
        data = self._read(self.path)
        self._get(data['cases'], key)
        del data['cases'][key]
        write_json(self.path, data)

    def save_run(self, case, answer, model, error=None):
        data = self._read(self.runs_path)
        key = uuid4().hex[:16]
        record = {'id': key, 'case_id': case['id'], 'case_snapshot': deepcopy(case),
                  'answer': answer, 'model': model, 'created_at': now(),
                  'status': 'error' if error else 'ok', 'error': error, 'rating': None}
        data['runs'][key] = record
        write_json(self.runs_path, data)
        return record

    def run(self, key):
        return self._get(self._read(self.runs_path)['runs'], key)

    def rate(self, key, rating):
        if rating not in {'correct', 'wrong', 'not_found'}:
            raise ValueError('Неизвестная оценка.')
        data = self._read(self.runs_path)
        record = self._get(data['runs'], key)
        if record['status'] != 'ok':
            raise ValueError('Ошибка выполнения не оценивается как качество поиска.')
        record.update(rating=rating, rated_at=now())
        data['runs'][key] = record
        write_json(self.runs_path, data)

    def latest(self):
        latest = {}
        for run in self._read(self.runs_path)['runs'].values():
            latest[run['case_id']] = run
        return latest

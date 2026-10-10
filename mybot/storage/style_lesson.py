"""Resumable lessons and a fixed source-disjoint holdout, scoped to a workspace."""

import random
import re
from uuid import uuid4

from mybot.storage.atomic import read_json, write_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, candidates, now
from mybot.services.conversation_boundaries import gap_seconds, wait_review_candidates
from mybot.services.conversation_boundaries import crossing_reason


class StyleLesson:
    def __init__(self, workspace):
        self.workspace = workspace
        self.store = PracticeStore(workspace)
        self.root = workspace.root / 'learning'

    def read(self, name):
        data = read_json(self.root / name)
        if data is not None and (data.get('account_id'), data.get('dialog_id')) != (
                self.workspace.account_id, self.workspace.dialog_id):
            raise ValueError('Данные занятия принадлежат другому диалогу.')
        return data

    def write(self, name, data):
        data.update(account_id=self.workspace.account_id, dialog_id=self.workspace.dialog_id)
        write_json(self.root / name, data)

    def holdout_ref(self):
        active = self.read('holdout_active.json')
        return active['name'] if active else 'holdout.json'

    def holdout_for(self, name):
        if name != 'holdout.json' and not re.fullmatch(r'holdout_[a-f0-9]{32}\.json', name):
            raise ValueError('Некорректная версия проверочной выборки.')
        data = self.read(name)
        if data:
            for key in data['cases']:
                self.store.source(self.store.get(key))
        return data

    def holdout(self, *, create=False):
        data = self.holdout_for(self.holdout_ref())
        if data:
            return data
        if not create:
            return None
        # Do not reserve previously shown or used training sources.
        used = {i for case in self.store.records() for i in case['context_ids'] + case['expected_ids']}
        for path in (self.root / 'reviewed').glob('*.json'):
            manifest = read_json(path)
            used.update(i for item in manifest.get('examples', {}).get('train', []) for i in item['ids'])
        for path in (self.root / 'datasets').glob('*/sources.json'):
            source = read_json(path)
            used.update(i for item in source.get('train', []) for i in item['context_ids'] + item['response_ids'])
        options = candidates(self.store.history(), load_deleted_message_ids(self.workspace.deleted_message_ids), set(),
                             threshold=gap_seconds(self.workspace))
        chosen = []
        for context, expected in reversed(options):
            ids = {m['message_id'] for m in context + expected}
            if not used.intersection(ids):
                chosen.append((context, expected))
                used.update(ids)
                if len(chosen) == 5:
                    break
        if len(chosen) < 2:
            raise ValueError('Не хватает независимых отрывков для проверки. Нужна более длинная история.')
        data = {'version': 1, 'cases': [self.store.create_from(c, e, 'holdout')['id'] for c, e in chosen],
                'created_at': now()}
        self.write('holdout.json', data)
        return data

    def holdout_audit(self, name=None):
        data = self.holdout_for(name or self.holdout_ref())
        if not data:
            return []
        messages = {m['message_id']: m for m in self.store.history()}
        deleted = load_deleted_message_ids(self.workspace.deleted_message_ids)
        return [{'case_id': key, 'reason': crossing_reason(*self.store.source(self.store.get(key),
                messages=messages, deleted=deleted), gap_seconds(self.workspace))}
                for key in data['cases']]

    def replacement_preview(self):
        history = self.store.history()
        options = candidates(history, load_deleted_message_ids(self.workspace.deleted_message_ids), set(),
                             threshold=gap_seconds(self.workspace))
        old = {key for item in self.holdout_audit() for key in
               self.store.get(item['case_id'])['context_ids'] + self.store.get(item['case_id'])['expected_ids']}
        chosen, used = [], set()
        for context, expected in reversed(options):
            ids = {m['message_id'] for m in context + expected}
            if not ids.intersection(old | used):
                chosen.append((context, expected))
                used.update(ids)
                if len(chosen) == 5:
                    break
        if len(chosen) < 2:
            raise ValueError('Не хватает независимых корректных ситуаций для новой проверки. Обновите историю.')
        return chosen

    def save_replacement_preview(self):
        from mybot.storage.reply_practice import digest
        chosen = self.replacement_preview()
        self.write('holdout_replacement_preview.json', {
            'status': 'previewed', 'old_ref': self.holdout_ref(),
            'sources': [{'context_ids': [m['message_id'] for m in context],
                         'expected_ids': [m['message_id'] for m in expected],
                         'digest': digest(context + expected)} for context, expected in chosen]})
        return chosen

    def replace_holdout(self):
        from mybot.storage.reply_practice import digest
        preview = self.read('holdout_replacement_preview.json')
        if not preview or preview.get('status') != 'previewed' or preview.get('old_ref') != self.holdout_ref():
            raise ValueError('Сначала откройте предпросмотр новой проверочной выборки.')
        chosen = self.replacement_preview()
        actual = [{'context_ids': [m['message_id'] for m in context],
                   'expected_ids': [m['message_id'] for m in expected],
                   'digest': digest(context + expected)} for context, expected in chosen]
        if actual != preview['sources']:
            raise ValueError('История изменилась после предпросмотра. Откройте предпросмотр заново.')
        name = f'holdout_{uuid4().hex}.json'
        data = {'version': 2, 'cases': [self.store.create_from(c, e, 'holdout')['id'] for c, e in chosen],
                'created_at': now(), 'boundary_gap_seconds': gap_seconds(self.workspace)}
        self.write(name, data)
        self.write('holdout_active.json', {'name': name})
        preview['status'] = 'applied'
        self.write('holdout_replacement_preview.json', preview)
        return data

    def reserved_ids(self):
        holdout = self.holdout()
        from mybot.services.reply_evaluation import reserved as evaluation_reserved
        return evaluation_reserved(self.workspace) | {i for key in (holdout or {}).get('cases', [])
                for case in [self.store.get(key)] for i in case['context_ids'] + case['expected_ids']}

    def current(self):
        return self.read('lesson.json')

    def start(self):
        self.holdout(create=True)
        state = self.current()
        if state and state['status'] != 'complete':
            state['status'] = 'active'
            self.write('lesson.json', state)
            return state
        records = self.store.records()
        used = {case['expected_ids'][0] for case in records if case['expected_ids']}
        used_wait = {case['context_ids'][-1] for case in records if not case['expected_ids']}
        excluded = self.reserved_ids()
        cutoff = min(excluded)
        options = [(c, e) for c, e in candidates(self.store.history(),
            load_deleted_message_ids(self.workspace.deleted_message_ids), set(),
            threshold=gap_seconds(self.workspace))
            if max(m['message_id'] for m in c + e) < cutoff
            and not excluded.intersection(m['message_id'] for m in c + e)]
        if not options:
            raise ValueError('Новых отрывков для занятия нет. Обновите историю или обучите выбранные примеры.')
        from mybot.services.history_sampling import boundaries, period
        cuts = boundaries([e[0]['message_id'] for c, e in options])
        sequence = (state or {}).get('sequence', -1) + 1
        options = [{'context': c, 'expected': e, 'period': period(e[0]['message_id'], cuts)}
                   for c, e in options if e[0]['message_id'] not in used]
        if not options:
            raise ValueError('Новых отрывков для занятия нет.')
        waiting = [(c, evidence) for c, evidence in wait_review_candidates(self.store.history(),
                   load_deleted_message_ids(self.workspace.deleted_message_ids),
                   threshold=gap_seconds(self.workspace))
                   if c[-1]['message_id'] not in used_wait
                   and max(m['message_id'] for m in c) < cutoff
                   and not excluded.intersection(m['message_id'] for m in c)]
        # Review suspected pauses as questions, not as labels inferred from silence.
        waiting = random.sample(waiting, min(2, len(waiting)))
        quotas = [3, 3, 3]
        quotas[(sequence + 2) % 3] += 1
        for context, _ in waiting:
            quotas[period(context[-1]['message_id'], cuts)] -= 1
        buckets = [[item for item in options if item['period'] == number] for number in range(3)]
        for bucket in buckets:
            random.shuffle(bucket)
        selected = []
        for bucket, quota in zip(buckets, quotas):
            selected.extend(bucket[:quota])
            del bucket[:quota]
        while len(selected) + len(waiting) < 10 and any(buckets):
            for bucket in buckets:
                if bucket and len(selected) + len(waiting) < 10:
                    selected.append(bucket.pop())
        cases = []
        for item in selected:
            case = self.store.create_from(item['context'], item['expected'])
            case.update(period=item['period'], date=item['expected'][0].get('date'), variant_count=2)
            write_json(self.store.path(case['id']), case)
            cases.append(case['id'])
        for context, evidence in waiting:
            case = self.store.create_from(context, [], 'wait_review')
            case.update(period=period(context[-1]['message_id'], cuts), date=context[-1].get('date'),
                        variant_count=2, boundary_evidence=evidence, action_kind='unconfirmed')
            write_json(self.store.path(case['id']), case)
            cases.append(case['id'])
        random.shuffle(cases)
        state = {'id': uuid4().hex, 'cases': cases, 'index': 0, 'choices': {},
                 'status': 'active', 'created_at': now(), 'sequence': sequence}
        self.write('lesson.json', state)
        return state

    def require(self, key, *, validate=True):
        state = self.current()
        if not state or state['status'] != 'active' or state['cases'][state['index']] != key:
            raise ValueError('Карточка устарела. Продолжите текущее занятие.')
        if validate:
            self.store.source(self.store.get(key))
        return state

    def advance(self, key, choice):
        state = self.require(key, validate=choice != 'skip')
        state.pop('input_case', None)
        state['choices'][key] = choice
        state['index'] += 1
        if state['index'] == len(state['cases']):
            state['status'] = 'complete'
        self.write('lesson.json', state)
        return state

    def pause(self):
        state = self.current()
        if state and state['status'] == 'active':
            state.pop('input_case', None)
            state['status'] = 'paused'
            self.write('lesson.json', state)

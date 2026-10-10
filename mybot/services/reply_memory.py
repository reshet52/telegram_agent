"""Resumable CPU evidence index and versioned, source-grounded dialog facts.

Raw utterances are evidence, not automatically extracted factual assertions.
Only the understanding step can extract facts, and every fact must quote a real
message. Drafts and lesson confirmations never enter the factual store.
"""

import json
import re
import sqlite3
from contextlib import contextmanager, closing

from mybot.services.conversation_boundaries import gap_seconds, boundary
from mybot.storage.atomic import read_json
from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore, candidates, digest, now, source_message


def terms(text):
    stop = {'the', 'and', 'you', 'your', 'are', 'это', 'что', 'как', 'для', 'меня'}
    return list(dict.fromkeys(w for w in re.findall(r'\w+', text.casefold())
                             if len(w) >= 3 and w not in stop))[:24]


class ReplyMemory:
    def __init__(self, workspace):
        self.ws = workspace
        self.path = workspace.root / 'reply_assistant/evidence.sqlite3'

    @contextmanager
    def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as db, db:
            db.row_factory = sqlite3.Row
            db.executescript('''
                CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
                CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, body TEXT,
                    hash TEXT, reason TEXT, active INTEGER NOT NULL DEFAULT 1);
                CREATE VIRTUAL TABLE IF NOT EXISTS evidence USING fts5(text, tokenize='porter unicode61');
                CREATE TABLE IF NOT EXISTS facts(id TEXT PRIMARY KEY, subject TEXT, property TEXT,
                    quote TEXT, source_id INTEGER, source_hash TEXT, recorded_at TEXT);
                CREATE TABLE IF NOT EXISTS revisions(message_id INTEGER, hash TEXT, body TEXT,
                    changed_at TEXT, PRIMARY KEY(message_id, hash));
            ''')
            identity = json.dumps([self.ws.account_id, self.ws.dialog_id])
            saved = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            if saved and saved[0] != identity:
                raise ValueError('Индекс памяти принадлежит другому диалогу.')
            if not saved and (db.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
                              or db.execute('SELECT COUNT(*) FROM facts').fetchone()[0]):
                raise ValueError('Индекс потерял сведения о принадлежности. Требуется явное восстановление.')
            db.execute("INSERT OR IGNORE INTO meta VALUES('identity', ?)", (identity,))
            yield db

    def sync(self, *, messages=None, batch_size=500, progress=None):
        """Reconcile each source by hash; committed batches survive interruption."""
        messages = PracticeStore(self.ws).history() if messages is None else messages
        ids = {m['message_id'] for m in messages}
        deleted = load_deleted_message_ids(self.ws.deleted_message_ids)
        with self.connection() as db:
            checkpoint = db.execute("SELECT value FROM meta WHERE key='checkpoint'").fetchone()
            if checkpoint and int(checkpoint[0]) not in ids | deleted:
                raise ValueError('Последний источник индекса отсутствует. Требуется явное восстановление.')
            for offset in range(0, len(messages), batch_size):
                for m in messages[offset:offset + batch_size]:
                    key, hashed = m['message_id'], digest([m])
                    reason = ('deleted' if key in deleted else
                              'media_without_verified_text' if m.get('type') not in (None, 'text') else
                              'empty_text' if not isinstance(m.get('text'), str) or not m['text'].strip() else None)
                    old = db.execute('SELECT * FROM messages WHERE id=?', (key,)).fetchone()
                    if old and old['hash'] == hashed and old['reason'] == reason and old['active']:
                        continue
                    if old:
                        db.execute('INSERT OR IGNORE INTO revisions VALUES(?,?,?,?)',
                                   (key, old['hash'], old['body'], now()))
                    db.execute('INSERT OR REPLACE INTO messages VALUES(?,?,?,?,1)',
                               (key, json.dumps(m, ensure_ascii=False), hashed, reason))
                    db.execute('DELETE FROM evidence WHERE rowid=?', (key,))
                    if reason is None:
                        db.execute('INSERT INTO evidence(rowid,text) VALUES(?,?)', (key, m['text']))
                if messages[offset:offset + batch_size]:
                    db.execute("INSERT OR REPLACE INTO meta VALUES('checkpoint', ?)",
                               (str(messages[min(offset + batch_size, len(messages)) - 1]['message_id']),))
                db.commit()
                if progress:
                    progress(min(offset + batch_size, len(messages)), len(messages))
            missing = [r[0] for r in db.execute('SELECT id FROM messages WHERE active=1') if r[0] not in ids]
            for key in missing:
                db.execute('UPDATE messages SET active=0, reason=? WHERE id=?', ('source_missing', key))
                db.execute('DELETE FROM evidence WHERE rowid=?', (key,))
            counts = {r[0] or 'indexed': r[1] for r in db.execute(
                'SELECT reason,COUNT(*) FROM messages WHERE active=1 GROUP BY reason')}
            db.execute("INSERT OR REPLACE INTO meta VALUES('updated_at', ?)", (now(),))
            return {'total': len(messages), 'counts': counts, 'missing': len(missing)}

    def sync_if_changed(self):
        """Skip full reconciliation only when history and deletions are unchanged."""
        def revision():
            result = []
            for path in (self.ws.chat_history, self.ws.deleted_message_ids):
                if not path.exists():
                    result.append(None)
                else:
                    stat = path.stat()
                    result.append([stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])
            return json.dumps(result)
        before = revision()
        with self.connection() as db:
            saved = db.execute("SELECT value FROM meta WHERE key='source_revision'").fetchone()
            if saved and saved[0] == before:
                return
        self.sync()
        if before == revision():
            with self.connection() as db:
                db.execute("INSERT OR REPLACE INTO meta VALUES('source_revision', ?)", (before,))

    def search(self, query, cutoff, *, limit=12, exclude=()):
        words = terms(query)
        if not words:
            return []
        expression = ' OR '.join('"' + word + '"' for word in words)
        excluded = set(exclude)
        with self.connection() as db:
            rows = db.execute('''SELECT m.body FROM evidence JOIN messages m ON m.id=evidence.rowid
                WHERE evidence MATCH ? AND m.id<=? AND m.active=1 AND m.reason IS NULL
                ORDER BY bm25(evidence),m.id DESC LIMIT ?''', (expression, cutoff, limit * 8))
            return [m for row in rows if (m := json.loads(row[0]))['message_id'] not in excluded][:limit]

    def facts(self, query, cutoff, *, limit=8):
        wanted = set(terms(query))
        with self.connection() as db:
            rows = list(db.execute('''SELECT f.*,m.body FROM facts f JOIN messages m ON m.id=f.source_id
                WHERE m.active=1 AND m.reason IS NULL AND m.hash=f.source_hash AND f.source_id<=?
                ORDER BY f.source_id DESC''', (cutoff,)))
            matching = {(r['subject'], r['property']) for r in rows
                        if wanted.intersection(terms(r['quote'] + ' ' + r['property']))}
            latest, selected = set(), []
            for row in rows:
                key = (row['subject'], row['property'])
                if key in latest:
                    continue
                latest.add(key)
                if key in matching:
                    fact = dict(row)
                    fact['source_date'] = json.loads(fact.pop('body')).get('date')
                    selected.append(fact)
                    if len(selected) == limit:
                        break
            return selected

    def remember(self, facts, sources):
        sources = {m['message_id']: source_message(m) for m in sources}
        with self.connection() as db:
            for fact in facts:
                if type(fact.get('source_id')) is not int:
                    continue
                m = sources.get(fact.get('source_id'))
                quote = fact.get('quote')
                if (not m or not isinstance(quote, str) or len(quote.strip()) < 3
                        or quote not in (m.get('text') or '')
                        or fact.get('subject') not in {'owner', 'person', 'situation'}
                        or (fact['subject'] == 'owner' and m['sender'] != 'Я')
                        or (fact['subject'] == 'person' and m['sender'] == 'Я')):
                    continue
                prop = fact.get('property')
                if not isinstance(prop, str) or not 1 <= len(prop) <= 100:
                    continue
                hashed = digest([m])
                row = db.execute('SELECT hash,reason,active FROM messages WHERE id=?', (m['message_id'],)).fetchone()
                if not row or row['hash'] != hashed or row['reason'] or not row['active']:
                    continue
                key = digest([fact['subject'], prop.casefold(), quote, m['message_id'], hashed])
                db.execute('INSERT OR IGNORE INTO facts VALUES(?,?,?,?,?,?,?)',
                    (key, fact['subject'], prop.casefold(), quote, m['message_id'], hashed, now()))


def retrieve(workspace, context, instruction, *, historical=False, extra_queries=()):
    """All evidence is from this dialog and at/before the immutable incoming cutoff."""
    from mybot.services.online_dataset import reserved_ids
    memory = ReplyMemory(workspace)
    memory.sync_if_changed()
    cutoff = max((m['message_id'] for m in context), default=0)
    query = '\n'.join([instruction] + [m.get('text') or '' for m in context[-5:]] + list(extra_queries))
    evidence = memory.search(query, cutoff)
    history = [m for m in PracticeStore(workspace).history() if m['message_id'] <= cutoff]
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    facts = memory.facts(query, cutoff)
    by_id = {m['message_id']: m for m in history}
    positions = {m['message_id']: i for i, m in enumerate(history)}
    evidence += [by_id[f['source_id']] for f in facts if f['source_id'] in by_id]
    expanded, seen = [], set()
    for hit in evidence:
        index = positions.get(hit['message_id'])
        if index is None:
            continue
        neighbors = [hit]
        for offset in (-1, 1):
            if 0 <= index + offset < len(history):
                neighbor = history[index + offset]
                pair = (neighbor, hit) if offset < 0 else (hit, neighbor)
                if (neighbor['message_id'] not in deleted and neighbor.get('type') in (None, 'text')
                        and neighbor.get('text') and not boundary(*pair, gap_seconds(workspace))):
                    neighbors.append(neighbor)
        for message in sorted(neighbors, key=lambda m: m['message_id']):
            if message['message_id'] not in seen:
                expanded.append(message)
                seen.add(message['message_id'])
    evidence = expanded
    reserved = reserved_ids(workspace)
    options = candidates(history, deleted, set(), threshold=gap_seconds(workspace))
    # Confirmed replacements override raw targets; they are style evidence, never facts.
    corrections = {}
    from mybot.services.practice_learning import chosen_reply
    store = PracticeStore(workspace)
    cases = store.records()
    for case in cases:
        ids = case['context_ids'] + case['expected_ids']
        if not case['expected_ids'] or max(ids) > cutoff or set(ids) & (reserved | deleted):
            continue
        try:
            c, e = store.source(case, messages=by_id, deleted=deleted)
        except ValueError:
            continue
        choice = chosen_reply(case, e)
        if not choice:
            continue
        confirmed_at = max([case.get('learning_updated_at', case['created_at']),
                            (case.get('owner_correction') or {}).get('at', '')]
                           + [r.get('at', '') for r in case.get('ratings', {}).values()])
        if historical and confirmed_at > str(context[-1].get('date') or ''):
            continue
        corrections[tuple(case['expected_ids'])] = {'text': choice[1], 'confirmed_at': confirmed_at}
    for path in (workspace.root / 'learning/confirmed').glob('*.json'):
        review = read_json(path, {})
        if (review.get('account_id'), review.get('dialog_id')) != (workspace.account_id, workspace.dialog_id):
            raise ValueError('Подтверждённый пример принадлежит другому диалогу.')
        ids = review.get('context_ids', []) + review.get('response_ids', [])
        if not ids or max(ids) > cutoff or set(ids) & (reserved | deleted):
            continue
        if historical and review.get('confirmed_at', '') > str(context[-1].get('date') or ''):
            continue
        case = PracticeStore(workspace).get(review['case_id'])
        try:
            PracticeStore(workspace).source(case)
        except ValueError:
            continue
        key = tuple(review['response_ids'])
        if key not in corrections or review['confirmed_at'] > corrections[key]['confirmed_at']:
            corrections[key] = review
    wanted, ranked = set(terms(query)), []
    for c, e in options:
        ids = [m['message_id'] for m in c + e]
        if set(ids) & reserved:
            continue
        if any(case.get('owner_action', {}).get('action') == 'wait'
               and case['context_ids'] == [m['message_id'] for m in c]
               and (not historical or case['owner_action']['at'] <= str(context[-1].get('date') or ''))
               for case in cases):
            continue
        incoming = []
        for m in reversed(c):
            if m['sender'] == 'Я':
                break
            incoming.append(m['text'])
        trigger = '\n'.join(reversed(incoming))
        overlap = wanted.intersection(terms(trigger))
        if not overlap:
            continue
        review = corrections.get(tuple(m['message_id'] for m in e))
        text = review['text'] if review else '\n\n'.join(m['text'] for m in e)
        item = {'incoming': trigger, 'reply': text, 'source_ids': ids,
                'source_digest': digest(c + e),
                'source_kind': 'owner_confirmed_reply' if review else 'historical_sent_reply'}
        recent = history[len(history) * 2 // 3]['message_id'] if history else cutoff
        ranked.append((len(overlap) + (2 if review else 0) + (.5 if max(ids) >= recent else 0), max(ids), item))
    ranked.sort(key=lambda v: (v[0], v[1]), reverse=True)
    examples, used = [], 0
    for _, _, item in ranked:
        size = len(json.dumps(item, ensure_ascii=False))
        if len(examples) < 3 and used + size <= 2800:
            examples.append(item)
            used += size
    # Preserve source integrity while bounding whole-message evidence.
    selected, used = [], 0
    for m in evidence:
        size = len(json.dumps(m, ensure_ascii=False))
        if used + size <= 5000:
            selected.append(m)
            used += size
    actions = []
    for case in cases:
        action = case.get('owner_action', {})
        ids = case['context_ids'] + case['expected_ids']
        if (action.get('action') != 'wait' or not ids or max(ids) > cutoff
                or set(ids) & (reserved | deleted)
                or (historical and action.get('at', '') > str(context[-1].get('date') or ''))):
            continue
        try:
            prior, _ = store.source(case, messages=by_id, deleted=deleted)
        except ValueError:
            continue
        trigger = '\n'.join(m['text'] for m in prior[-4:])
        if set(terms(query)).intersection(terms(trigger)):
            actions.append({'action': 'wait', 'context': prior[-4:], 'source_ids': ids,
                            'source_digest': case['source_digest'], 'confirmed_at': action['at']})
    return {'cutoff_id': cutoff, 'evidence': selected, 'facts': facts,
            'style_examples': examples, 'action_examples': actions[:3]}

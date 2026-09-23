"""Per-dialog evidence for future correction learning; no inferred acceptance."""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from uuid import uuid4


SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    generation_id TEXT PRIMARY KEY,
    account_id INTEGER NOT NULL,
    dialog_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    context_message_id INTEGER,
    model TEXT NOT NULL,
    variants_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outgoing_replies (
    message_id INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL,
    dialog_id INTEGER NOT NULL,
    sent_at TEXT,
    text TEXT,
    message_type TEXT NOT NULL,
    generation_id TEXT,
    chosen_variant INTEGER,
    result TEXT,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);
CREATE TABLE IF NOT EXISTS reply_drafts (
    draft_id TEXT PRIMARY KEY,
    account_id INTEGER NOT NULL,
    dialog_id INTEGER NOT NULL,
    generation_id TEXT,
    chosen_variant INTEGER,
    original_text TEXT,
    final_text TEXT NOT NULL,
    context_message_id INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    telegram_message_id INTEGER,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);
CREATE TABLE IF NOT EXISTS generation_rejections (
    generation_id TEXT PRIMARY KEY,
    rejected_at TEXT NOT NULL,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);
CREATE TABLE IF NOT EXISTS draft_message_links (
    message_id INTEGER PRIMARY KEY,
    draft_id TEXT NOT NULL,
    linked_at TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES outgoing_replies(message_id),
    FOREIGN KEY (draft_id) REFERENCES reply_drafts(draft_id)
);
CREATE TABLE IF NOT EXISTS draft_feedback (
    draft_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    result TEXT,
    finalized_at TEXT,
    FOREIGN KEY (draft_id) REFERENCES reply_drafts(draft_id)
);
"""


def _open(workspace):
    workspace.root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(workspace.reply_journal)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
    except BaseException:
        connection.close()
        raise
    return connection


def record_generation(workspace, variants, context_message_id, model):
    """Persist candidates before the control bot presents them to the owner."""
    if not variants:
        raise ValueError("No generated variants to record")
    generation_id = str(uuid4())
    candidates = [
        {"variant_id": f"{generation_id}:{number}", "text": text}
        for number, text in enumerate(variants, 1)
    ]
    with closing(_open(workspace)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO generations VALUES (?, ?, ?, ?, ?, ?, ?)",
                (generation_id, workspace.account_id, workspace.dialog_id,
                 datetime.now(timezone.utc).isoformat(), context_message_id, model,
                 json.dumps(candidates, ensure_ascii=False)),
            )
    return generation_id


def record_outgoing_reply(workspace, message):
    """Record actual sending once by Telegram message ID, without guessing a choice."""
    record_outgoing_replies(workspace, [message])


def record_outgoing_replies(workspace, messages):
    """Backfill after restart and deduplicate live/catch-up deliveries."""
    rows = [
        (message["message_id"], workspace.account_id, workspace.dialog_id,
         message.get("date"), message.get("text"),
         message.get("type") or "unknown")
        for message in messages
        if message.get("message_id") is not None and message.get("sender") == "Я"
    ]
    if not rows:
        return
    with closing(_open(workspace)) as connection:
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO outgoing_replies "
                "(message_id, account_id, dialog_id, sent_at, text, message_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )


def get_generation(workspace, generation_id):
    with closing(_open(workspace)) as connection:
        row = connection.execute(
            "SELECT context_message_id, variants_json FROM generations "
            "WHERE generation_id=? AND account_id=? AND dialog_id=?",
            (generation_id, workspace.account_id, workspace.dialog_id),
        ).fetchone()
    if row is None:
        return None
    return {"context_message_id": row[0], "variants": json.loads(row[1])}


def create_draft(workspace, generation_id, variant_number):
    generation = get_generation(workspace, generation_id)
    if generation is None:
        raise ValueError("Вариант относится к другому или недоступному диалогу.")
    with closing(_open(workspace)) as connection:
        if connection.execute("SELECT 1 FROM generation_rejections WHERE generation_id=?",
                              (generation_id,)).fetchone():
            raise ValueError("Эта генерация уже отклонена.")
        if connection.execute(
            "SELECT 1 FROM reply_drafts WHERE generation_id=? "
            "AND status IN ('sending', 'sent', 'uncertain') LIMIT 1",
            (generation_id,),
        ).fetchone():
            raise ValueError("Ответ из этой генерации уже отправлен или проверяется.")
    variant = next((item for item in generation["variants"]
                    if item["variant_id"] == f"{generation_id}:{variant_number}"), None)
    if variant is None:
        raise ValueError("Вариант не найден.")
    return _insert_draft(workspace, generation_id, variant_number, variant["text"],
                         generation["context_message_id"])


def _insert_draft(workspace, generation_id, variant_number, text, context_message_id):
    draft_id = str(uuid4())
    with closing(_open(workspace)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO reply_drafts "
                "(draft_id, account_id, dialog_id, generation_id, chosen_variant, "
                "original_text, final_text, context_message_id, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)",
                (draft_id, workspace.account_id, workspace.dialog_id,
                 generation_id, variant_number, text, text, context_message_id,
                 datetime.now(timezone.utc).isoformat()),
            )
    return draft_id


def get_draft(workspace, draft_id):
    with closing(_open(workspace)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM reply_drafts WHERE draft_id=? AND account_id=? AND dialog_id=?",
            (draft_id, workspace.account_id, workspace.dialog_id),
        ).fetchone()
    return dict(row) if row else None


def get_outgoing_reply(workspace, message_id):
    with closing(_open(workspace)) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM outgoing_replies WHERE message_id=? AND account_id=? AND dialog_id=?",
            (message_id, workspace.account_id, workspace.dialog_id),
        ).fetchone()
    return dict(row) if row else None


def recent_kept_drafts(workspace, limit=5):
    with closing(_open(workspace)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM reply_drafts WHERE account_id=? AND dialog_id=? AND status='kept' "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (workspace.account_id, workspace.dialog_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def link_outgoing_to_draft(workspace, message_id, draft_id):
    """An explicit owner choice; leave its result pending until the group is complete."""
    with closing(_open(workspace)) as connection:
        with connection:
            draft = connection.execute(
                "SELECT generation_id, chosen_variant FROM reply_drafts "
                "WHERE draft_id=? AND account_id=? AND dialog_id=? AND status='kept'",
                (draft_id, workspace.account_id, workspace.dialog_id),
            ).fetchone()
            outgoing = connection.execute(
                "SELECT text, result FROM outgoing_replies "
                "WHERE message_id=? AND account_id=? AND dialog_id=?",
                (message_id, workspace.account_id, workspace.dialog_id),
            ).fetchone()
            if draft is None or outgoing is None or not outgoing[0]:
                raise ValueError("Черновик или текст исходящего сообщения недоступен.")
            prior = connection.execute(
                "SELECT draft_id FROM draft_message_links WHERE message_id=?", (message_id,)
            ).fetchone()
            if prior and prior[0] != draft_id:
                raise ValueError("Сообщение уже привязано к другому варианту.")
            if outgoing[1] == 'independent_reply':
                raise ValueError("Сообщение уже отмечено как самостоятельное.")
            connection.execute(
                "INSERT OR IGNORE INTO draft_message_links VALUES (?, ?, ?)",
                (message_id, draft_id, datetime.now(timezone.utc).isoformat()),
            )
            connection.execute(
                "INSERT INTO draft_feedback (draft_id, status) VALUES (?, 'open') "
                "ON CONFLICT(draft_id) DO UPDATE SET status='open', result=NULL, finalized_at=NULL",
                (draft_id,),
            )
            # Adding another segment reopens the whole answer: no stale acceptance label.
            connection.execute(
                "UPDATE outgoing_replies SET generation_id=?, chosen_variant=?, result='linked_pending' "
                "WHERE message_id IN (SELECT message_id FROM draft_message_links WHERE draft_id=?)",
                (draft[0], draft[1], draft_id),
            )
    return len(linked_outgoing_replies(workspace, draft_id))


def linked_outgoing_replies(workspace, draft_id):
    with closing(_open(workspace)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT o.* FROM outgoing_replies o JOIN draft_message_links l "
            "ON o.message_id=l.message_id WHERE l.draft_id=? AND o.account_id=? "
            "AND o.dialog_id=? ORDER BY o.message_id",
            (draft_id, workspace.account_id, workspace.dialog_id),
        ).fetchall()
    return [dict(row) for row in rows]


def finalize_draft_feedback(workspace, draft_id):
    """Compare the explicitly grouped sent text with the original suggestion."""
    draft = get_draft(workspace, draft_id)
    if draft is None or draft['status'] != 'kept':
        raise ValueError("Оставленный вариант недоступен.")
    outgoing = linked_outgoing_replies(workspace, draft_id)
    if not outgoing:
        raise ValueError("Сначала привяжите хотя бы одно исходящее сообщение.")
    actual = "\n\n".join(row['text'].strip() for row in outgoing)
    result = ('accepted_without_edit' if actual == draft['original_text'].strip()
              else 'accepted_with_edit')
    with closing(_open(workspace)) as connection:
        with connection:
            connection.execute(
                "INSERT INTO draft_feedback VALUES (?, 'finalized', ?, ?) "
                "ON CONFLICT(draft_id) DO UPDATE SET status='finalized', result=excluded.result, "
                "finalized_at=excluded.finalized_at",
                (draft_id, result, datetime.now(timezone.utc).isoformat()),
            )
            connection.execute(
                "UPDATE outgoing_replies SET result=? WHERE message_id IN "
                "(SELECT message_id FROM draft_message_links WHERE draft_id=?)",
                (result, draft_id),
            )
    return result, actual, len(outgoing)


def mark_independent_reply(workspace, message_id):
    with closing(_open(workspace)) as connection:
        with connection:
            if connection.execute(
                "SELECT 1 FROM draft_message_links WHERE message_id=?", (message_id,)
            ).fetchone():
                raise ValueError("Сообщение уже привязано к варианту.")
            changed = connection.execute(
                "UPDATE outgoing_replies SET result='independent_reply' "
                "WHERE message_id=? AND account_id=? AND dialog_id=? AND text IS NOT NULL "
                "AND (result IS NULL OR result='independent_reply')",
                (message_id, workspace.account_id, workspace.dialog_id),
            ).rowcount
    if not changed:
        raise ValueError("Исходящее сообщение недоступно.")


def update_draft_text(workspace, draft_id, text):
    with closing(_open(workspace)) as connection:
        with connection:
            changed = connection.execute(
                "UPDATE reply_drafts SET final_text=? WHERE draft_id=? AND "
                "account_id=? AND dialog_id=? AND status='ready'",
                (text, draft_id, workspace.account_id, workspace.dialog_id),
            ).rowcount
    return changed == 1


def cancel_draft(workspace, draft_id):
    with closing(_open(workspace)) as connection:
        with connection:
            changed = connection.execute(
                "UPDATE reply_drafts SET status='cancelled' WHERE draft_id=? AND "
                "account_id=? AND dialog_id=? AND status='ready'",
                (draft_id, workspace.account_id, workspace.dialog_id),
            ).rowcount
    return changed == 1


def mark_draft_kept(workspace, draft_id):
    """The owner kept text for copying; this says nothing about actual sending."""
    with closing(_open(workspace)) as connection:
        with connection:
            changed = connection.execute(
                "UPDATE reply_drafts SET status='kept' WHERE draft_id=? AND "
                "account_id=? AND dialog_id=? AND status='ready'",
                (draft_id, workspace.account_id, workspace.dialog_id),
            ).rowcount
    return changed == 1


def reject_generation(workspace, generation_id):
    if get_generation(workspace, generation_id) is None:
        raise ValueError("Генерация недоступна для этого диалога.")
    with closing(_open(workspace)) as connection:
        with connection:
            if connection.execute(
                "SELECT 1 FROM reply_drafts WHERE generation_id=? "
                "AND status IN ('sending', 'sent', 'uncertain') LIMIT 1",
                (generation_id,),
            ).fetchone():
                raise ValueError("Ответ из этой генерации уже отправлен или проверяется.")
            connection.execute(
                "INSERT OR IGNORE INTO generation_rejections VALUES (?, ?)",
                (generation_id, datetime.now(timezone.utc).isoformat()),
            )


def unreject_generation(workspace, generation_id):
    if get_generation(workspace, generation_id) is None:
        return False
    with closing(_open(workspace)) as connection:
        with connection:
            connection.execute("DELETE FROM generation_rejections WHERE generation_id=?",
                               (generation_id,))
    return True

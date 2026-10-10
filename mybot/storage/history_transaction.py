"""Stage a dialog update; rollback an interrupted multi-file publication on startup."""

import hashlib
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from mybot.storage.atomic import read_json, write_json


FILES = ('chat_history.jsonl', 'episodes.jsonl', 'episode_embeddings.json', 'episode_embeddings_live.jsonl',
         'episode_pending_batch.json')
MARKER = '.history_update_pending.json'


def digest(path):
    if not path.exists():
        return None
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def copy_atomic(source, target):
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def recover_history_update(workspace):
    marker = workspace.root / MARKER
    pending = read_json(marker)
    if not pending:
        return
    name = pending['directory']
    if Path(name).name != name or not name.startswith('.history-update-'):
        raise ValueError('Некорректный журнал обновления истории.')
    directory = workspace.root / name
    for filename in FILES:
        if filename not in pending['original']:
            continue  # Transactions written before base embeddings were staged.
        original = pending['original'][filename]
        backup = directory / 'before' / filename
        if original is not None and digest(backup) != original:
            raise RuntimeError('Повреждена резервная копия обновления истории; восстановление остановлено.')
    for filename in FILES:
        if filename not in pending['original']:
            continue
        target = workspace.root / filename
        if pending['original'][filename] is None:
            target.unlink(missing_ok=True)
        else:
            copy_atomic(directory / 'before' / filename, target)
    marker.unlink()
    shutil.rmtree(directory)


class HistoryTransaction:
    """Caller holds the live-message lock from snapshot through publication.

    publish() deliberately has no await: cancellation is accepted before commit,
    never between replacements. A durable undo journal covers process failure.
    """

    def __init__(self, workspace):
        self.workspace = workspace
        recover_history_update(workspace)
        self.directory = Path(tempfile.mkdtemp(prefix='.history-update-', dir=workspace.root))
        self.staged = replace(workspace, root=self.directory / 'after')
        self.staged.root.mkdir()
        (self.directory / 'before').mkdir()
        self.original = {}
        try:
            for name in (*FILES, 'deleted_message_ids.json'):
                source = workspace.root / name
                self.original[name] = digest(source)
                if source.exists():
                    copy_atomic(source, self.directory / 'before' / name)
                    copy_atomic(source, self.staged.root / name)
        except BaseException:
            self.close()
            raise

    def publish(self):
        if any(digest(self.workspace.root / name) != value for name, value in self.original.items()):
            raise RuntimeError('История изменилась во время обновления. Повторите операцию.')
        marker = self.workspace.root / MARKER
        write_json(marker, {'directory': self.directory.name, 'original': self.original})
        try:
            for name in FILES:
                source = self.staged.root / name
                if digest(source) != self.original[name]:
                    if not source.exists():
                        raise RuntimeError('Обновление не должно удалять рабочие файлы.')
                    copy_atomic(source, self.workspace.root / name)
        except BaseException:
            recover_history_update(self.workspace)
            raise
        marker.unlink()

    def close(self):
        # Preserve undo data if publication or rollback was interrupted.
        if not (self.workspace.root / MARKER).exists():
            shutil.rmtree(self.directory, ignore_errors=True)

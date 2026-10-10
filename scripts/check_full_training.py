"""Bounded real-model smoke test on synthetic text; never reads a real dialog."""

import json
import os
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from mybot.services.local_training import MODEL, PROJECT, PYTHON
from mybot.services.full_history_dataset import prepare_full
from mybot.storage.atomic import read_json, write_json
from mybot.storage.workspace import Workspace


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='mybot-full-check-') as temporary:
        ws = Workspace(1, 1, 'Synthetic', Path(temporary) / 'data/account_1/dialogs/dialog_1')
        write_json(ws.manifest, dict(account_id=1, dialog_id=1, dialog_name='Synthetic'))
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        messages = [dict(message_id=i + 1, sender='Other' if i % 2 == 0 else 'Я', type='text',
                         date=(start + timedelta(hours=(i // 2) * 5, seconds=i % 2)).isoformat(),
                         text=f'Coffee at {i % 10 + 8} tomorrow?' if i % 2 == 0 else 'yes, see you tomorrow')
                    for i in range(1000)]
        ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
        manifest = prepare_full(ws, tokenizer, MODEL)
        key = uuid4().hex
        root = ws.root / 'learning/mlx_runs' / key
        root.mkdir(parents=True)
        for steps, resuming in ((2, False), (3, True)):
            command = [str(PYTHON), '-m', 'scripts.train_reply_mlx', '--full', '--account', '1', '--dialog', '1',
                       '--dataset', manifest['id'], '--model-path', str(MODEL), '--run-id', key,
                       '--hours', '.2', '--max-steps', str(steps), '--skip-comparison']
            if resuming:
                command.append('--resume')
            with (root / 'process.log').open('a') as log:
                process = subprocess.Popen(command, cwd=temporary,
                    env=dict(os.environ, PYTHONPATH=str(PROJECT), HF_HUB_OFFLINE='1',
                             HF_HUB_DISABLE_TELEMETRY='1'), stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise RuntimeError('Synthetic training exceeded five minutes')
            if code:
                raise RuntimeError((root / 'process.log').read_text()[-6000:])
            run = read_json(root / 'run.json')
            assert run['completed_steps'] == steps, run
            assert (root / 'checkpoint.json').is_file()
            assert not (ws.root / 'learning/selected_model.json').exists()
            print({k: run.get(k) for k in ('completed_steps', 'loss_before', 'train_loss', 'peak_memory_gb',
                                         'thermal_monitor', 'stop_reason')}, flush=True)
        print('Synthetic full preparation, real Qwen training and resume: OK', flush=True)


if __name__ == '__main__':
    main()

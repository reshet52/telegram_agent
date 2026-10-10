"""Explicit bounded MLX integration check with synthetic history only.

Run with .venv-mlx/bin/python -m scripts.check_style_pipeline --steps 5.
No Telegram connection, owner dataset or production model selection is modified.
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from uuid import uuid4

from mybot.storage.workspace import Workspace
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.atomic import read_json, write_json
from mybot.services.reviewed_dataset import prepare_reviewed
from mybot.services import style_versions, style_comparison, training_jobs
from mybot.ai import local_reply


async def check(steps):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mybot-style-check-') as temporary:
        workspace = Workspace(1, 1, 'Synthetic test', Path(temporary) / 'data/account_1/dialogs/dialog_1')
        write_json(workspace.root / 'workspace.json', {'account_id': 1, 'dialog_id': 1, 'dialog_name': 'Synthetic test'})
        messages = []
        activities = ['coffee', 'a walk', 'lunch', 'a movie', 'dinner']
        for i in range(1, 601):
            text = (f'Want to meet for {activities[(i // 2) % 5]} tomorrow?' if i % 2
                    else f'Sounds good, see you at {8 + (i // 2) % 10} :)')
            messages.append({'message_id': i, 'sender': 'Other' if i % 2 else 'Я',
                             'text': text, 'type': 'text'})
        workspace.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
        lesson = StyleLesson(workspace)
        state = lesson.start()
        # Synthetic owner-approved reference answers; no fabricated quality scores.
        for key in state['cases']:
            case = lesson.store.get(key)
            case['use_original'] = True
            write_json(lesson.store.path(key), case)
            lesson.advance(key, 'use')
        manifest = prepare_reviewed(workspace)
        async def launch(key, iterations):
            root = workspace.root / 'learning/mlx_runs' / key
            root.mkdir(parents=True)
            with (root / 'process.log').open('w') as log:
                process = await asyncio.create_subprocess_exec(str(local_reply.PYTHON), '-m', 'scripts.train_reply_mlx',
                    '--account', '1', '--dialog', '1', '--dataset', manifest['id'],
                    '--model-path', str(local_reply.MODEL), '--steps', str(iterations), '--run-id', key,
                    cwd=temporary, env=dict(os.environ, PYTHONPATH=str(local_reply.PROJECT), HF_HUB_OFFLINE='1'),
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
            return process, root

        version = uuid4().hex
        process, root = await launch(version, steps)
        try:
            await asyncio.wait_for(process.wait(), 300)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode:
            raise RuntimeError((root / 'process.log').read_text())
        run = read_json(root / 'run.json')
        assert run['weights_before'] != run['weights_after']
        # Verify cancellation and process identity against an actual owned worker.
        cancelled_id = uuid4().hex
        process, cancelled_root = await launch(cancelled_id, 50)
        lesson.write('training_job.json', {'id': cancelled_id, 'pid': process.pid, 'state': 'preparing'})
        try:
            for _ in range(100):
                if (cancelled_root / 'run.json').is_file():
                    break
                await asyncio.sleep(.1)
            assert training_jobs.alive({'id': cancelled_id, 'pid': process.pid})
            training_jobs.stop(workspace)
            await asyncio.wait_for(process.wait(), 20)
            assert training_jobs.status(workspace)['state'] == 'stopped'
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        before = await style_comparison.generate_pair(workspace, version, 0)
        pair = before['pairs']['0']
        assert pair['candidate']['request']['receipt']['adapter_sha256'] == run['adapter_sha256']
        # Exercise the activation gate without declaring an artificial quality win.
        for index in range(1, len(before['cases'])):
            await style_comparison.generate_pair(workspace, version, index)
        for index in range(len(before['cases'])):
            style_comparison.vote(workspace, version, index, 'neither')
        style_versions.activate(workspace, version)
        request = local_reply.snapshot('Someone asks: Can we meet for coffee tomorrow? Reply briefly in English.',
                                       workspace, style_versions.selected(workspace))
        variants = await local_reply.generate(request)
        assert len(variants) == len(set(variants)) == 3
        assert request['receipt']['adapter_sha256'] == run['adapter_sha256']
        style_versions.rollback(workspace)
        assert style_versions.selected(workspace) == 'configured'
        report = {'synthetic_only': True, 'steps': steps, 'trainable_parameters': run['trainable_parameters'],
                  'loss_before': run['loss_before'], 'loss_after': run['loss_after'],
                  'base_reply_seconds': pair['baseline']['request']['receipt']['seconds'],
                  'adapted_reply_seconds': pair['candidate']['request']['receipt']['seconds'],
                  'three_variants_seconds': request['receipt']['seconds'],
                  'three_variants_total_seconds': request['receipt']['total_seconds'],
                  'elapsed_seconds': time.monotonic() - started,
                  'adapter_receipt_verified': True, 'rollback_verified': True,
                  'worker_cancellation_verified': True,
                  'quality_assessed': False}
        write_json(Path('local_models/style_pipeline_check.json'), report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 50:
        parser.error('Use 1..50 steps for this synthetic check.')
    asyncio.run(check(args.steps))

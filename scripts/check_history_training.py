"""Bounded real MLX bootstrap/refinement check on synthetic ordinary conversation."""

import asyncio
import json
import os
import tempfile
import time
import random
from pathlib import Path
from uuid import uuid4

from mybot.ai import local_reply
from mybot.services.history_training import prepare
from mybot.services import style_versions
from mybot.services.reply_practice import generate_practice
from mybot.storage.atomic import read_json, write_json
from mybot.storage.style_lesson import StyleLesson
from mybot.storage.workspace import Workspace


async def check():
    random.seed(42)
    started = time.monotonic()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local_reply.MODEL), local_files_only=True)
    with tempfile.TemporaryDirectory(prefix='mybot-bootstrap-') as temporary:
        ws = Workspace(1, 1, 'Synthetic', Path(temporary) / 'data/account_1/dialogs/dialog_1')
        write_json(ws.root / 'workspace.json', dict(account_id=1, dialog_id=1, dialog_name='Synthetic'))
        messages = [dict(message_id=i, sender='Я' if i % 2 == 0 else 'Other', type='text',
                         text=f'Coffee at {8 + i % 10} tomorrow?' if i % 2 else f'yes, see you at {8 + (i - 1) % 10}')
                    for i in range(1, 601)]
        ws.chat_history.write_text(''.join(json.dumps(m) + '\n\n' for m in messages))
        lesson = StyleLesson(ws)
        state = lesson.start()
        async def train(manifest):
            version = uuid4().hex
            root = ws.root / 'learning/mlx_runs' / version
            root.mkdir(parents=True)
            with (root / 'process.log').open('w') as log:
                process = await asyncio.create_subprocess_exec(str(local_reply.PYTHON), '-m', 'scripts.train_reply_mlx',
                    '--account', '1', '--dialog', '1', '--dataset', manifest['id'],
                    '--model-path', str(local_reply.MODEL), '--run-id', version,
                    cwd=temporary, env=dict(os.environ, PYTHONPATH=str(local_reply.PROJECT), HF_HUB_OFFLINE='1'),
                    stdout=log, stderr=asyncio.subprocess.STDOUT)
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
            assert run['completed_steps'] == manifest['counts']['train']
            if style_versions.practice_selected(ws) != version:
                quality = read_json(style_versions.quality_path(ws, version), {})
                raise RuntimeError({'gate': quality.get('status'), 'counts': quality.get('counts'),
                    'valid_actions': quality.get('valid_actions'),
                    'expected_actions': quality.get('expected_actions'),
                    'rejections': [a.get('rejection') for item in quality.get('action_results', [])
                                   for a in item.get('attempts', []) if a.get('rejection')]})
            assert style_versions.selected(ws) == 'configured'
            assert lesson.current() == state
            return version, run

        manifest = prepare(ws, tokenizer, local_reply.MODEL, limit=8)
        version, run = await train(manifest)
        from mybot.services.style_comparison import generate_pair
        pair = (await generate_pair(ws, version, 0))['pairs']['0']
        assert pair['candidate']['request']['receipt']['adapter_sha256'] == run['adapter_sha256']
        key = state['cases'][0]
        try:
            case = await generate_practice(lesson.store, key, local=True)
        except ValueError:
            case = lesson.store.get(key)
            assert not case['variants']
            receipt = (case.get('request') or case.get('action_request') or {}).get('receipt', {})
            assert receipt.get('adapter_sha256') == run['adapter_sha256']
            assert any(a['rejection'] for a in receipt.get('attempts', []))
        if case.get('model_action', {}).get('action') == 'wait':
            lesson.store.confirm_action(key, 'reply')
            try:
                case = await generate_practice(lesson.store, key, local=True)
            except ValueError:
                case = lesson.store.get(key)
        if case.get('request'):
            assert case['request']['version'] == version
            assert case['request']['receipt']['adapter_sha256'] == run['adapter_sha256']
        if case['variants']:
            lesson.store.save_correction(key, 'yes, tomorrow works for me')
        else:
            lesson.store.confirm_action(key, 'reply')
        lesson.store.clear_learning(key)
        lesson.store.confirm_action(key, 'wait')
        refined = prepare(ws, tokenizer, local_reply.MODEL, purpose='refine', limit=8)
        assert any(item['row']['completion'][0]['content'] == 'WAIT'
                   for item in refined['examples']['train'])
        refined_version, refined_run = await train(refined)
        assert refined_run['weights_before'] == run['weights_after']
        assert refined_run['parent_sha256'] == run['adapter_sha256']
        request = local_reply.snapshot('Coffee tomorrow?', ws, refined_version, count=1)
        await local_reply.generate(request)
        assert request['receipt']['adapter_sha256'] == refined_run['adapter_sha256']
        if case['variants']:
            assert (await generate_practice(lesson.store, key, local=True))['request']['version'] == version
        else:
            assert (case.get('request') or case.get('action_request'))['version'] == version
        receipt = (case.get('request') or case.get('action_request'))['receipt']
        report = {'synthetic_only': True, 'quality_assessed': False,
            'bootstrap_steps': run['completed_steps'], 'refine_steps': refined_run['completed_steps'],
            'bootstrap_seconds': run['elapsed_seconds'], 'refine_seconds': refined_run['elapsed_seconds'],
            'bootstrap_loss': [run['loss_before'], run['loss_after']],
            'refine_loss': [refined_run['loss_before'], refined_run['loss_after']],
            'base_reply_seconds': pair['baseline']['request']['receipt']['seconds'],
            'adapted_reply_seconds': pair['candidate']['request']['receipt']['seconds'],
            'lesson_reply_count': case.get('variant_count', 3),
            'lesson_reply_seconds': receipt['total_seconds'],
            'lesson_reply_rejected': not bool(case['variants']),
            'bootstrap_hash': run['adapter_sha256'], 'refine_hash': refined_run['adapter_sha256'],
            'wait_action_trained': True,
            'parent_weights_loaded': True, 'practice_receipt_verified': True,
            'live_pointer_unchanged': True, 'lesson_preserved': True,
            'elapsed_seconds': time.monotonic() - started}
        write_json(Path('local_models/history_training_check.json'), report)
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(check())

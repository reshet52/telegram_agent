"""Opt-in real MLX checkpoint test: MYBOT_TEST_MLX=1, no downloaded model needed."""

import os
import random
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.environ.get('MYBOT_TEST_MLX') == '1', 'Explicit Metal verification only')
class CheckpointTests(unittest.TestCase):
    def test_online_preemption_and_resume_match_one_uninterrupted_epoch(self):
        import mlx.core as mx
        import mlx.nn as nn
        import numpy as np
        from unittest.mock import patch
        from mybot.services.full_local_training import train_full, file_hash
        from mybot.services.night_training_control import NightGuard, BudgetExpired
        from mybot.services.learning_dataset import example
        from mybot.storage.atomic import write_json, read_json
        from mybot.storage.workspace import Workspace
        from mybot.ai.local_format import FORMAT_VERSION

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(128, 4)
                self.layers = [nn.Linear(4, 4)]
                self.head = nn.Linear(4, 128)
            def __call__(self, tokens):
                return self.head(self.layers[0](self.embed(tokens)))

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return ''.join(m['content'] for m in messages) + '<|im_start|>assistant\n'
            def encode(self, text, **kwargs):
                return [ord(c) % 128 for c in text]

        class Interrupt(NightGuard):
            def __init__(self, at):
                super().__init__(1800, reader=lambda: 0)
                self.calls, self.at = 0, at
            def wait(self, training=True):
                self.calls += 1
                if self.calls == self.at:
                    self.stopping = True
                return super().wait(training)

        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace(1, 2, 'synthetic', Path(tmp) / 'dialog')
            base, parent = Path(tmp) / 'base', Path(tmp) / 'parent'
            base.mkdir()
            parent.mkdir()
            mx.random.seed(7)
            initial = Tiny()
            initial.save_weights(str(base / 'model.safetensors'))
            initial.save_weights(str(parent / 'adapters.safetensors'))
            write_json(base / 'config.json', {})
            fingerprint = file_hash(parent / 'adapters.safetensors')
            write_json(parent / 'run.json', {'adapter_sha256': fingerprint, 'base_model': str(base.resolve())})
            rows = {'train': [example([{'sender': 'Other', 'text': f'q{i}'}], [{'text': f'a{i}'}]) for i in range(3)],
                    'valid': [example([{'sender': 'Other', 'text': 'q'}], [{'text': 'a'}])]}
            manifest = {'version': 4, 'id': 'a'*32, 'hash': 'immutable', 'parent_adapter': 'b'*32,
                        'parent_sha256': fingerprint, 'format_version': FORMAT_VERSION,
                        'tokenizer_hash': 'tokenizer', 'max_length': 512, 'order': [2, 0, 1],
                        'base_steps': 3, 'counts': {'train': 3, 'valid': 1}, 'coverage_counts': {}}
            def load(*args, **kwargs):
                model = Tiny()
                model.load_weights(str(base / 'model.safetensors'))
                return model, Tokenizer()
            with patch('mybot.services.full_local_training.load_reviewed', return_value=(manifest, rows)), patch(
                    'mybot.services.full_local_training.tokenizer_hash', return_value='tokenizer'), patch(
                    'mybot.services.style_versions.adapter', return_value=parent), patch(
                    'mybot.services.style_versions.selected', return_value='b'*32), patch(
                    'mlx_lm.load', side_effect=load), patch('mlx_lm.tuner.utils.linear_to_lora_layers',
                                                         side_effect=lambda m, *a: m.unfreeze()), patch(
                    'mlx_lm.tuner.trainer.grad_checkpoint'):
                complete = train_full(ws, manifest['id'], base, run_id='1'*32, skip_comparison=True,
                                      guard=NightGuard(1800, reader=lambda: 0))
                for at, key in ((3, '2'*32), (5, '3'*32)):
                    paused = train_full(ws, manifest['id'], base, run_id=key, skip_comparison=True, guard=Interrupt(at))
                    self.assertEqual(read_json(paused / 'run.json')['status'], 'paused')
                    resumed = train_full(ws, manifest['id'], base, run_id=key, resume=True,
                                         skip_comparison=True, guard=NightGuard(1800, reader=lambda: 0))
                    self.assertEqual(read_json(resumed / 'run.json')['completed_steps'], 3)
                    def weights(root, name):
                        pointer = read_json(root / 'checkpoint.json')
                        return mx.load(str(root / 'checkpoints' / pointer['folder'] / name))
                    for filename in ('adapters.safetensors', 'optimizer.npz', 'random.npz'):
                        expected, actual = weights(complete, filename), weights(resumed, filename)
                        self.assertEqual(set(expected), set(actual))
                        for name in expected:
                            np.testing.assert_array_equal(np.array(expected[name]), np.array(actual[name]))

    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        import numpy as np
        from mlx.utils import tree_flatten, tree_unflatten
        from mybot.services.full_local_training import save_checkpoint, restore_checkpoint
        mx.random.seed(42)
        random.seed(42)
        np.random.seed(42)
        model = nn.Linear(2, 2)
        optimizer = optim.Adam(learning_rate=2e-5)
        loss_grad = nn.value_and_grad(model, lambda m: mx.sum(m(mx.ones((1, 2))) ** 2))
        def step():
            loss, gradient = loss_grad(model)
            optimizer.update(model, gradient)
            mx.eval(model.parameters(), optimizer.state, loss)
        for _ in range(3):
            step()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_checkpoint(root, model, optimizer, {'completed_steps': 3, 'order': [2, 0, 1]},
                            mx=mx, tree_flatten=tree_flatten)
            expected_rng = (random.random(), np.random.random(), np.array(mx.random.uniform(shape=(4,))))
            step()
            step()
            expected = [(name, np.array(tensor)) for name, tensor in tree_flatten(model.parameters())]
            restored = restore_checkpoint(root, model, optimizer, mx=mx, tree_unflatten=tree_unflatten)
            self.assertEqual(restored, {'completed_steps': 3, 'order': [2, 0, 1]})
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(np.random.random(), expected_rng[1])
            np.testing.assert_array_equal(np.array(mx.random.uniform(shape=(4,))), expected_rng[2])
            step()
            step()
            self.assertEqual(optimizer.step.item(), 5)
            for (name, actual), (other, desired) in zip(tree_flatten(model.parameters()), expected):
                self.assertEqual(name, other)
                np.testing.assert_array_equal(np.array(actual), desired)
            pointer = __import__('json').loads((root / 'checkpoint.json').read_text())
            (root / 'checkpoints' / pointer['folder'] / 'optimizer.npz').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                restore_checkpoint(root, model, optimizer, mx=mx, tree_unflatten=tree_unflatten)

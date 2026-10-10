import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.ai.local_format import FORMAT_VERSION, prompt_tokens, training_tokens, rejection
from mybot.ai.local_sampling import sample
from tests.test_history_training import Tokenizer


class LocalFormatTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_partial_generation_preserves_only_accepted_text(self):
        from mybot.ai import local_reply
        result = {'variants': ['Hey❤️'], 'adapter_sha256': None, 'model': 'test', 'seconds': 1,
                  'format_version': FORMAT_VERSION,
                  'attempts': [{'text': 'Hey❤️', 'rejection': None}, {'text': '<think>bad', 'rejection': 'service_tokens'}],
                  'error': 'invalid_generation'}
        process = NS(returncode=0, communicate=AsyncMock(return_value=(json.dumps(result).encode(), b'')))
        request = {'count': 2, 'expected_adapter_sha256': None, 'allow_partial': True}
        with patch('mybot.ai.local_reply.available', return_value=True), \
             patch('mybot.ai.local_worker.worker.exchange', AsyncMock(return_value=result)):
            self.assertEqual(await local_reply.generate(request), ['Hey❤️'])
        self.assertTrue(request['receipt']['partial_generation'])

    def test_only_final_answer_and_eos_are_supervised(self):
        tokenizer = Tokenizer()
        row = {'prompt': [{'role': 'user', 'content': 'hello'}],
               'completion': [{'role': 'assistant', 'content': 'Привет!'}]}
        tokens, offset = training_tokens(tokenizer, row)
        self.assertEqual(tokens[:offset], prompt_tokens(tokenizer, row['prompt']))
        self.assertEqual(bytes(tokens[offset:]).decode(), 'Привет!<|im_end|>')
        self.assertNotIn(b'<think>', bytes(tokens[offset:]))
        row['completion'][0]['content'] = '<think>bad</think>'
        with self.assertRaises(ValueError):
            training_tokens(tokenizer, row)

    def test_invalid_outputs_and_normal_short_style(self):
        for text in ('<think>bad', 'hello</think>', '<|im_end|>'):
            self.assertEqual(rejection(text, [1], 'stop'), 'service_tokens')
        self.assertEqual(rejection('', [], 'stop'), 'empty')
        self.assertEqual(rejection('unfinished', [1], 'length'), 'token_limit')
        for width in range(1, 5):
            self.assertEqual(rejection('loop', list(range(width)) * 6, 'stop'), 'repetition')
        self.assertEqual(rejection('dominant', [1, 1, 2] * 11, 'stop'), 'repetition')
        ids = [1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 7, 1, 1, 8, 1, 1, 9, 1, 1, 10, 1, 1, 11, 1, 1]
        self.assertEqual(rejection('dominant', ids, 'stop'), 'dominant_token')
        for text in ('Ok❤️', 'yes yes', 'ha ha ha', 'Some new expression'):
            self.assertIsNone(rejection(text, list(range(len(text))), 'stop'))

    def test_sampling_has_bounded_retries_and_preserves_rejections(self):
        tokenizer = Tokenizer()
        tokenizer.eos_token_ids = {99}
        calls = []
        def stream(*args, **kwargs):
            calls.append(kwargs)
            yield NS(text='<think>bad', token=7, finish_reason='stop')
        core = NS(random=NS(seed=Mock()))
        modules = {'mlx': NS(core=core), 'mlx.core': core,
                   'mlx_lm': NS(stream_generate=stream), 'mlx_lm.sample_utils': NS(make_sampler=Mock())}
        with patch.dict(sys.modules, modules):
            variants, attempts = sample(None, tokenizer, [{'role': 'user', 'content': 'Hi'}], 42, count=3)
        self.assertEqual(variants, [])
        self.assertEqual(len(calls), 6)
        self.assertEqual([a['seed'] for a in attempts], list(range(42, 48)))
        self.assertTrue(all(a['rejection'] == 'service_tokens' for a in attempts))

    async def test_worker_error_is_not_a_variant_or_api_fallback(self):
        from mybot.ai import local_reply
        result = {'variants': [], 'adapter_sha256': None, 'model': 'test', 'seconds': 1,
                  'format_version': FORMAT_VERSION, 'attempts': [{'rejection': 'repetition'}],
                  'error': 'invalid_generation'}
        process = NS(returncode=0, communicate=AsyncMock(return_value=(json.dumps(result).encode(), b'')))
        request = {'count': 3, 'expected_adapter_sha256': None}
        with patch('mybot.ai.local_reply.available', return_value=True), \
             patch('mybot.ai.local_worker.worker.exchange', AsyncMock(return_value=result)):
            with self.assertRaisesRegex(ValueError, 'Попытки исчерпаны'):
                await local_reply.generate(request)
        self.assertEqual(request['receipt']['attempts'][0]['rejection'], 'repetition')

    async def test_duplicate_failure_has_specific_message(self):
        from mybot.ai import local_reply
        result = {'variants': ['Ok'], 'adapter_sha256': None, 'model': 'test', 'seconds': 1,
                  'format_version': FORMAT_VERSION, 'attempts': [{'rejection': None}, {'rejection': 'duplicate'}],
                  'error': 'invalid_generation'}
        process = NS(returncode=0, communicate=AsyncMock(return_value=(json.dumps(result).encode(), b'')))
        with patch('mybot.ai.local_reply.available', return_value=True), \
             patch('mybot.ai.local_worker.worker.exchange', AsyncMock(return_value=result)):
            with self.assertRaisesRegex(ValueError, 'три разных варианта'):
                await local_reply.generate({'count': 3, 'expected_adapter_sha256': None})

"""Model routing checks without paid API requests or real credentials."""
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from mybot.ai import client
from mybot.config import Config


class ModelTests(unittest.IsolatedAsyncioTestCase):
    def test_defaults_and_explicit_overrides(self):
        path = Path(__file__).resolve().parents[1] / 'mybot/config.py'
        environment = {'API_ID': '1', 'OPENAI_API_KEY': 'offline-test'}
        with patch.dict(os.environ, environment, clear=True), patch('dotenv.load_dotenv'):
            config = runpy.run_path(str(path))['Config']
            self.assertEqual(config.OPENAI_MODEL, 'gpt-6-luna')
            self.assertEqual(config.ANALYSIS_MODEL, 'gpt-6-luna')
            self.assertEqual(config.EMBEDDING_MODEL, 'text-embedding-3-small')
            os.environ['OPENAI_MODEL'] = 'custom-reply'
            os.environ['ANALYSIS_MODEL'] = 'custom-analysis'
            config = runpy.run_path(str(path))['Config']
            self.assertEqual(config.OPENAI_MODEL, 'custom-reply')
            self.assertEqual(config.ANALYSIS_MODEL, 'custom-analysis')

    async def test_reply_and_analysis_request_models(self):
        response = NS(output_text='{}', usage=NS(input_tokens=1, output_tokens=1, total_tokens=2))
        create = AsyncMock(return_value=response)
        with patch.object(Config, 'OPENAI_MODEL', 'gpt-6-luna'), \
             patch.object(Config, 'ANALYSIS_MODEL', 'gpt-6-luna'), \
             patch.object(client.ai_client.responses, 'create', create), patch('builtins.print'):
            await client.test_ai_connection()
            await client.generate_answers('test')
            await client.analyze_full_history('test')
            await client.merge_memory_category('user_style', [])
            await client.analyze_incremental_memory('test')
        self.assertEqual(create.await_count, 5)
        self.assertTrue(all(call.kwargs['model'] == 'gpt-6-luna' for call in create.await_args_list))

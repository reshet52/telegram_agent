import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock, patch

from mybot.ai.local_worker import LocalWorker
from mybot.ai.local_format import FORMAT_VERSION
from scripts.local_reply_mlx import Engine


class ResidentTests(unittest.IsolatedAsyncioTestCase):
    def test_same_dialog_reuses_weights_but_other_dialog_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            load = Mock(return_value=(object(), object()))
            sample = Mock(return_value=(['ok'], [{'seconds': .1}]))
            with patch('scripts.local_reply_mlx.ROOT', Path(tmp)), \
                 patch.dict('sys.modules', {'mlx_lm': NS(load=load)}), \
                 patch('mybot.ai.local_sampling.sample', sample):
                engine = Engine()
                request = dict(model=tmp, format_version=FORMAT_VERSION, count=1, messages=[], account_id=1, dialog_id=2)
                first = engine.execute(request)
                second = engine.execute(request)
                self.assertFalse(first['timings']['reused_model'])
                self.assertTrue(second['timings']['reused_model'])
                self.assertEqual(second['timings']['load_seconds'], 0)
                request['dialog_id'] = 3
                engine.execute(request)
                self.assertEqual(load.call_count, 2)
                engine.clear()
                self.assertIsNone(engine.lock)

    async def test_variant_is_forwarded_before_final_result(self):
        worker = LocalWorker()
        progress = {'event': 'variant', 'variants': ['first']}
        final = {'event': 'result', 'result': {'variants': ['first', 'second']}}
        worker.process = NS(stdin=NS(write=Mock(), drain=AsyncMock()),
                            stdout=NS(readline=AsyncMock(side_effect=[json.dumps(progress).encode(), json.dumps(final).encode()])))
        callback = AsyncMock()
        with patch.object(worker, 'start', AsyncMock()):
            result = await worker.exchange({}, callback)
        callback.assert_awaited_once_with(progress)
        self.assertEqual(result['variants'], ['first', 'second'])

    async def test_failure_stops_child_and_releases_busy_flag(self):
        worker = LocalWorker()
        worker.process = NS(returncode=None, stdin=NS(write=Mock(), drain=AsyncMock()),
                            stdout=NS(readline=AsyncMock(return_value=b'')), terminate=Mock(), wait=AsyncMock())
        process = worker.process
        with patch.object(worker, 'start', AsyncMock()), self.assertRaises(ValueError):
            await worker.exchange({})
        process.terminate.assert_called_once()
        process.wait.assert_awaited_once()
        self.assertFalse(worker.busy)
        self.assertIsNone(worker.process)

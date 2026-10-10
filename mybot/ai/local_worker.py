"""One reusable MLX child per controller, with bounded lifetime and no prompt cache."""

import asyncio
import json
import os
import time

from mybot.services.local_training import PROJECT, PYTHON
from mybot.storage.atomic import write_json


class LocalWorker:
    def __init__(self):
        self.process = None
        self.drain = None
        self.busy = False

    async def stop(self):
        process, self.process = self.process, None
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if self.drain:
            self.drain.cancel()
            await asyncio.gather(self.drain, return_exceptions=True)
            self.drain = None

    async def release(self):
        # Other controllers/benchmark processes can hold a resident model too.
        write_json(PROJECT / 'local_models/.inference_release.json', {'at': time.time()})
        if not self.busy:
            await self.stop()

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        await self.stop()
        # Ask an idle resident worker in another process to release the GPU.
        write_json(PROJECT / 'local_models/.inference_release.json', {'at': time.time()})
        self.process = await asyncio.create_subprocess_exec(str(PYTHON), '-m', 'scripts.local_reply_mlx', '--serve',
            cwd=PROJECT, env=dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1'),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=4*1024*1024)
        process = self.process
        async def drain():
            # Read/discard diagnostic stderr to avoid a full pipe. Never echo model text.
            while await process.stderr.read(8192):
                pass
        self.drain = asyncio.create_task(drain())

    async def exchange(self, request, on_variant=None):
        await self.start()
        self.busy = True
        try:
            self.process.stdin.write((json.dumps(request, ensure_ascii=False) + '\n').encode())
            await self.process.stdin.drain()
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise ValueError('Локальный процесс завершился. Можно повторить локальную часть.')
                value = json.loads(line)
                if value.get('event') == 'variant':
                    if on_variant:
                        await on_variant(value)
                elif value.get('event') == 'result':
                    if value.get('worker_error'):
                        raise ValueError('Локальная генерация не выполнена: ' + value['worker_error'])
                    return value['result']
                else:
                    raise ValueError('Некорректное сообщение локального процесса.')
        except BaseException:
            await self.stop()
            raise
        finally:
            self.busy = False


worker = LocalWorker()

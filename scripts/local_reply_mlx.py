"""Isolated MLX inference: one-shot compatibility or a reusable JSON-lines worker."""
import fcntl
import gc
import hashlib
import json
import os
import select
import sys
import time
from pathlib import Path
from mybot.ai.local_format import FORMAT_VERSION
from mybot.storage.atomic import read_json

ROOT = Path(__file__).resolve().parents[1] / 'local_models'

class Engine:
    def __init__(self):
        self.model = self.tokenizer = self.key = self.lock = None
        self.loaded_at = 0

    def clear(self):
        self.model = self.tokenizer = self.key = None
        gc.collect()
        if 'mlx.core' in sys.modules:
            sys.modules['mlx.core'].clear_cache()
        if self.lock:
            self.lock.close()
            self.lock = None

    def execute(self, request, emit=None):
        from mlx_lm import load
        from mybot.ai.local_sampling import sample
        from mybot.ai.local_reply import evaluation_preempted
        started = time.monotonic()
        model_path = Path(request['model']).resolve()
        fingerprint = (hashlib.sha256((Path(request['adapter']) / 'adapters.safetensors').read_bytes()).hexdigest()
                       if request.get('adapter') else None)
        if fingerprint != request.get('expected_adapter_sha256'):
            raise ValueError('adapter_mismatch')
        key = (str(model_path), request.get('adapter'), fingerprint,
               request.get('account_id'), request.get('dialog_id'), request.get('workspace_root'))
        timings = {'load_seconds': 0., 'gpu_wait_seconds': 0., 'reused_model': key == self.key}
        if key != self.key:
            self.clear()
            self.lock = (ROOT / '.training.lock').open('a')
            wait = time.monotonic()
            while True:
                try:
                    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - wait > 150:
                        raise ValueError('gpu_busy')
                    time.sleep(.2)
            timings['gpu_wait_seconds'] = time.monotonic() - wait
            before_load = time.monotonic()
            self.model, self.tokenizer = load(str(model_path), adapter_path=request.get('adapter'))
            timings['load_seconds'] = time.monotonic() - before_load
            self.key, self.loaded_at = key, time.time()
        if request.get('format_version') != FORMAT_VERSION:
            raise ValueError('format_mismatch')
        count = request.get('count', 3)
        task = request.get('task', 'reply')
        if count not in (1, 2, 3) or task not in ('reply', 'decision') or (task == 'decision' and count != 1):
            raise ValueError('invalid_request')
        scores = None
        if task == 'decision':
            from mybot.ai.action_scoring import score_actions
            action, scores = score_actions(self.model, self.tokenizer, request['messages'])
            variants, attempts = [action], []
        else:
            def progress(variants, attempts):
                if emit:
                    emit({'event': 'variant', 'variants': variants, 'adapter_sha256': fingerprint,
                          'format_version': FORMAT_VERSION, 'attempts': attempts, 'timings': timings})
            variants, attempts = sample(self.model, self.tokenizer, request['messages'], request.get('seed', 42),
                count=count, max_tokens=request.get('max_tokens', 384), diverse=request.get('diverse', False),
                should_stop=lambda: evaluation_preempted(request), on_variant=progress,
                reuse_prefix=request.get('reuse_prefix', False))
        timings['sampling_seconds'] = sum(a['seconds'] for a in attempts)
        return {'variants': variants, 'seconds': time.monotonic() - started,
                'format_version': FORMAT_VERSION, 'attempts': attempts, 'timings': timings,
                'decision_scores': scores, 'error': 'invalid_generation' if len(variants) != count else None,
                'adapter_sha256': fingerprint, 'model': str(model_path)}

def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    engine = Engine()
    def emit(value):
        print(json.dumps(value, ensure_ascii=False), flush=True)
    try:
        if '--serve' not in sys.argv:
            emit(engine.execute(json.load(sys.stdin)))
            return
        idle = time.monotonic()
        while True:
            release = read_json(ROOT / '.inference_release.json', {}).get('at', 0)
            if engine.key and (release > engine.loaded_at or time.monotonic() - idle >= 120):
                engine.clear()
            if time.monotonic() - idle >= 120:
                return
            if not select.select([sys.stdin], [], [], 1)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                return
            try:
                request = json.loads(line)
                result = engine.execute(request, emit)
                emit({'event': 'result', 'result': result})
            except Exception as error:
                engine.clear()
                emit({'event': 'result', 'worker_error': type(error).__name__})
            idle = time.monotonic()
    finally:
        engine.clear()

if __name__ == '__main__':
    main()

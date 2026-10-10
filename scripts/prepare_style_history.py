"""CPU preflight: validate and tokenize sources before asking to start GPU training."""

import argparse
import os
import re
import signal

from mybot.services.history_training import prepare, prepare_repair
from mybot.storage.atomic import write_json
from scripts.train_reply_adapter import resolve_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', required=True, type=int)
    parser.add_argument('--dialog', required=True, type=int)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--purpose', choices=['history', 'refine'], default='history')
    parser.add_argument('--repair-version')
    parser.add_argument('--full', action='store_true', help='Все пригодные ответы, отдельная новая проверка.')
    parser.add_argument('--max-length', type=int, default=2048)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-f0-9]{32}', args.run_id):
        parser.error('Invalid run ID')
    ws = resolve_workspace(args.account, args.dialog)
    path = ws.root / 'learning/mlx_runs' / args.run_id / 'run.json'
    run = dict(account_id=ws.account_id, dialog_id=ws.dialog_id, pid=os.getpid(),
               status='preparing', phase='preparing')
    write_json(path, run)
    def interrupted(*_):
        raise KeyboardInterrupt('Подготовка остановлена.')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        if args.full and (args.repair_version or args.purpose != 'history'):
            raise ValueError('Полный режим не совмещается с восстановлением/доработкой.')
        if args.full:
            from mybot.services.full_history_dataset import prepare_full
            manifest = prepare_full(ws, tokenizer, args.model_path, max_length=args.max_length)
        else:
            manifest = (prepare_repair(ws, tokenizer, args.model_path, args.repair_version) if args.repair_version
                    else prepare(ws, tokenizer, args.model_path, purpose=args.purpose))
        run.update(status='prepared', dataset_id=manifest['id'], summary={k: manifest[k] for k in
            ('purpose', 'counts', 'period_counts', 'reviewed_count', 'dates', 'estimated_minutes')})
        run['summary']['source_counts'] = manifest.get('source_counts', {})
        run['summary']['mode'] = manifest.get('mode', 'trial')
        run['summary']['coverage_counts'] = manifest.get('coverage_counts', {})
    except BaseException as error:
        run.update(status='interrupted_or_failed', error=str(error)[:400]
                   if isinstance(error, ValueError) else type(error).__name__)
        write_json(path, run)
        raise
    write_json(path, run)


if __name__ == '__main__':
    main()

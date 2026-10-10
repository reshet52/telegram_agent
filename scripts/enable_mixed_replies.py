"""Explicitly enable mixed replies and actual-sent learning in one workspace."""

import argparse

from mybot.services.online_learning import enable
from mybot.services.style_versions import activate
from mybot.storage.atomic import read_json, write_json
from scripts.train_reply_adapter import resolve_workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('--version', required=True, help='Version explicitly selected by the owner.')
    args = parser.parse_args()
    ws = resolve_workspace(args.account, args.dialog)
    smoke = read_json(ws.root / 'learning/mixed_smoke.json', {})
    if smoke.get('version') != args.version or not smoke.get('manual_relevance_checked'):
        raise ValueError('Сначала выполните сквозную проверку и проверьте уместность ответов.')
    activate(ws, args.version, explicit_owner=True)
    enable(ws)
    write_json(ws.root / 'learning/mixed_replies.json', {
        'enabled': True, 'account_id': ws.account_id, 'dialog_id': ws.dialog_id,
        'local_numbers': [1, 2], 'api_numbers': [3, 4], 'initial_version': args.version})
    print('Смешанные ответы и очередь фактически отправленных ответов включены для выбранного диалога.')


if __name__ == '__main__':
    main()

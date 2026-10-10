"""Small explicit API/local smoke test; no Telegram or automatic activation."""

import argparse
import asyncio
from types import SimpleNamespace

from mybot.services.mixed_replies import MixedReplies
from mybot.services.style_versions import adapter
from mybot.storage.atomic import write_json
from scripts.train_reply_adapter import resolve_workspace


async def check(ws, version, local_only=False):
    from unittest.mock import patch
    from mybot.services.reply_service import ReplyService
    adapter(ws, version)
    if local_only:
        from copy import deepcopy
        from mybot.ai import local_reply
        from mybot.storage.atomic import read_json
        path = ws.root / 'learning/mixed_smoke.json'
        report = read_json(path)
        for record in report['results']:
            state = record['sides']['local']
            request = deepcopy(state['request'])
            request['diverse'] = True
            request.pop('receipt', None)
            try:
                variants = await local_reply.generate(request)
                record['variants'] = [v for v in record['variants'] if v['source'] == 'api']
                record['variants'] += [{'number': i, 'source': 'local', 'text': v} for i, v in enumerate(variants, 1)]
                state.update(status='ready', request=request)
                state.pop('error', None)
            except ValueError as error:
                state.update(status='failed', request=request, error=str(error))
            print('Локальная часть:', state['status'], flush=True)
        write_json(path, report)
        print(path)
        return
    # Deliberately simple, synthetic situations: no private history is sent to the API.
    contexts = [
        ([{'message_id': -2, 'sender': 'Я', 'text': 'we can talk tomorrow', 'type': 'text'},
          {'message_id': -1, 'sender': 'Other', 'text': 'What time tomorrow? Would seven work for you?', 'type': 'text'}],
         'Ответь на последний вопрос: семь вечера подходит. Коротко, по-английски.'),
        ([{'message_id': -1, 'sender': 'Other', 'text': 'Can we talk tonight?', 'type': 'text'}],
         'Сегодня не могу. Предложи завтра после семи вечера. Ответь коротко по-английски; не соглашайся на сегодня.'),
    ]
    reports = []
    with patch('mybot.services.mixed_replies.selected', return_value=version), patch(
            'mybot.services.mixed_replies.build_local_context',
            side_effect=lambda c, instruction, *args: __import__('json').dumps({'instruction': instruction}, ensure_ascii=False)):
        for context, instruction in contexts:
            service = ReplyService(context, {}, workspace=ws)
            async def publish(record):
                pass
            result = await MixedReplies(service).start(instruction, publish)
            reports.append(result)
            print('Ситуация', len(reports), ':', {k: v['status'] for k, v in result['sides'].items()}, flush=True)
    path = ws.root / 'learning/mixed_smoke.json'
    write_json(path, {'version': version, 'results': reports, 'manual_relevance_checked': False})
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--account', type=int, required=True)
    parser.add_argument('--dialog', type=int, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    asyncio.run(check(resolve_workspace(args.account, args.dialog), args.version, args.local_only))


if __name__ == '__main__':
    main()

"""Same-dialog contextual examples from explicit owner feedback, never rules."""

from mybot.storage.deletions import load_deleted_message_ids
from mybot.storage.reply_practice import PracticeStore


def confirmed_examples(workspace, limit=100):
    store = PracticeStore(workspace)
    if not store.root.exists():
        return []
    # Read source history once even if many exercises have been reviewed.
    messages = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    result = []
    for case in store.records()[:limit]:
        if case.get('learning_disabled'):
            continue
        try:
            context, expected = store.source(case, messages=messages, deleted=deleted)
        except ValueError:
            continue
        correction = case.get('owner_correction')
        source_kind, text = None, None
        if correction:
            source_kind, text = 'owner_correction_not_sent', correction['text']
        elif case.get('use_original'):
            source_kind, text = 'historical_sent_reply', '\n\n'.join(m['text'] for m in expected)
        else:
            good = [(value['at'], int(number)) for number, value in case['ratings'].items()
                    if value['rating'] == 'good' and int(number) <= len(case['variants'])]
            if good:
                _, number = max(good)
                source_kind, text = 'approved_suggestion_not_sent', case['variants'][number - 1]
        if text:
            incoming = []
            for message in reversed(context):
                if message['sender'] == 'Я':
                    break
                incoming.append(message['text'])
            result.append({'generation_id': 'practice:' + case['id'],
                'trigger_text': '\n'.join(reversed(incoming)), 'original_text': '',
                'actual_text': text, 'result': source_kind, 'source_kind': source_kind})
    return result

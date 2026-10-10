"""A separate short action task for the same local adapter."""

from mybot.services.learning_dataset import example

SYSTEM = ('Реши, нужен ли сейчас ответ владельца аккаунта на последнюю реплику. '
          'Учитывай только переписку до этого момента. Прощание после уже состоявшегося '
          'прощания часто не требует ответа; вопрос или первое прощание может требовать. '
          'Верни ровно одно слово: REPLY или WAIT. Не добавляй объяснений.')


def messages(context):
    row = example(context, [])
    row['prompt'][0]['content'] = SYSTEM
    return row['prompt']


def training_row(context, action):
    if action not in {'reply', 'wait'}:
        raise ValueError('Неизвестное действие для обучения.')
    return {'prompt': messages(context),
            'completion': [{'role': 'assistant', 'content': action.upper()}]}


def confirmed_counts(workspace):
    """Human-confirmed confusion counts, never inferred from silent history."""
    from mybot.storage.reply_practice import PracticeStore
    from mybot.storage.deletions import load_deleted_message_ids
    counts = {'reviewed': 0, 'wrong_reply': 0, 'wrong_wait': 0}
    from mybot.services.conversation_boundaries import crossing_reason, gap_seconds
    store = PracticeStore(workspace)
    history = {m['message_id']: m for m in store.history()}
    deleted = load_deleted_message_ids(workspace.deleted_message_ids)
    for case in store.records():
        predicted = (case.get('model_action') or {}).get('action')
        actual = (case.get('owner_action') or {}).get('action')
        if predicted not in {'reply', 'wait'} or actual not in {'reply', 'wait'}:
            continue
        try:
            context, expected = store.source(case, messages=history, deleted=deleted)
        except ValueError:
            continue
        if expected and crossing_reason(context, expected, gap_seconds(workspace)):
            continue
        counts['reviewed'] += 1
        if predicted == 'reply' and actual == 'wait':
            counts['wrong_reply'] += 1
        if predicted == 'wait' and actual == 'reply':
            counts['wrong_wait'] += 1
    return counts

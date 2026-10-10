"""One bounded historical generation without retrieval or future-derived profiles."""

from mybot.storage.atomic import durable_call


async def generate_answers(*args, **kwargs):
    from mybot.ai.client import generate_answers as generate
    return await generate(*args, **kwargs)


async def build_ai_request(*args, **kwargs):
    from mybot.ai.context_builder import build_ai_request as build
    return await build(*args, **kwargs)


async def generate_practice(store, key, *, local=False):
    case = store.get(key)
    context, _ = store.source(case)
    if case['variants']:
        if local and case.get('request', {}).get('version') == 'base':
            raise ValueError('Сохранённые ответы базовой модели больше не используются в занятиях. Пропустите старую карточку.')
        return case
    if local:
        from mybot.ai import local_reply
        from mybot.services.decision_task import messages as decision_messages
        from mybot.services.style_versions import practice_selected
        from mybot.services.learning_dataset import example
        version = practice_selected(store.workspace)
        if version == 'base':
            raise ValueError('Для занятий нужна обученная версия, прошедшая автопроверку. '
                             'Откройте «Обучение и версии». Базовая модель вместо неё не используется.')
        if not case.get('model_action') and case.get('owner_action', {}).get('action') != 'reply':
            decision = local_reply.snapshot('', store.workspace, version,
                                            messages=decision_messages(context), count=1)
            decision['context_message_ids'] = case['context_ids']
            await durable_call(store.save_action_request, key, decision)
            try:
                action = await local_reply.decide(decision)
            finally:
                await durable_call(store.save_action_request, key, decision)
            case = await durable_call(store.save_decision, key, action, decision)
        if case.get('model_action', {}).get('action') == 'wait' and case.get('owner_action', {}).get('action') != 'reply':
            return case
        # The hidden completion is not passed to the worker, profiles or retrieval.
        request = local_reply.snapshot('', store.workspace, version,
                                       messages=example(context, [])['prompt'], count=case.get('variant_count', 3))
        request['context_message_ids'] = case['context_ids']
        await durable_call(store.save_request, key, request)
        try:
            variants = await local_reply.generate(request)
        finally:
            await durable_call(store.save_request, key, request)
        return await durable_call(store.save_result, key, variants)
    from mybot.ai.reply_provider import request_snapshot
    from mybot.services.reply_service import ReplyService
    prompt = await build_ai_request(context,
        'Историческая проверка: продолжи переписку от лица «Я» в стиле его предыдущих '
        'сообщений. Ты видишь только прошлое до скрытого ответа. Предложи три варианта '
        'следующего ответа, не ссылайся на последующие события.',
        {}, workspace=store.workspace, history_only=True, use_correction_examples=False)
    snapshot = request_snapshot(prompt)
    snapshot['context_message_ids'] = case['context_ids']
    await durable_call(store.save_request, key, snapshot)
    # No automatic retries here: a repeated click reuses a completed result.
    raw = await generate_answers(prompt)
    variants = ReplyService.split_variants(raw)
    return await durable_call(store.save_result, key, variants)

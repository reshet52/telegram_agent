"""One bounded historical generation without retrieval or future-derived profiles."""

from mybot.ai.client import generate_answers
from mybot.ai.context_builder import build_ai_request
from mybot.ai.reply_provider import request_snapshot
from mybot.services.reply_service import ReplyService
from mybot.storage.atomic import durable_call


async def generate_practice(store, key):
    case = store.get(key)
    context, _ = store.source(case)
    if case['variants']:
        return case
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

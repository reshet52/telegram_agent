"""Bounded, validated generation; raw rejected output stays in local receipts only."""

import time
from mybot.ai.local_format import prompt_tokens, rejection


def sample(model, tokenizer, messages, seed, *, count=1, attempts=None, max_tokens=384, accepted=None,
           should_stop=None, diverse=False, on_variant=None, reuse_prefix=False):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    prompt = prompt_tokens(tokenizer, messages)
    if len(prompt) > 8192:
        raise ValueError('Контекст слишком длинный для локальной модели (8192 токена).')
    variants, receipts = [], []
    prefix = None
    if reuse_prefix:
        from mybot.ai.prompt_prefix import PromptPrefix
        prefix = PromptPrefix(model)
    for index in range(attempts if attempts is not None else count * 2):
        if should_stop and should_stop():
            raise TimeoutError('Предел времени проверки достигнут.')
        started = time.monotonic()
        attempt_messages = messages
        attempt_prompt = prompt
        if diverse and variants:
            attempt_messages = [dict(m) for m in messages]
            attempt_messages[0]['content'] += ('\nПредложи другой естественный ответ на последнюю реплику, '
                'с тем же смыслом и указаниями владельца. Уже предложены: ' + repr(variants) +
                '. Не повторяй их дословно.')
            attempt_prompt = prompt_tokens(tokenizer, attempt_messages)
            if len(attempt_prompt) > 8192:
                raise ValueError('Контекст вариантов слишком длинный.')
        mx.random.seed(seed + index)
        parts, ids, reason = [], [], None
        suffix, cache_kwargs, reused = prefix.prepare(attempt_prompt) if prefix else (attempt_prompt, {}, 0)
        for response in stream_generate(model, tokenizer, prompt=suffix, max_tokens=max_tokens, **cache_kwargs,
                                        sampler=make_sampler(temp=0.8)):
            if should_stop and should_stop():
                raise TimeoutError('Предел времени проверки достигнут.')
            parts.append(response.text)
            ids.append(response.token)
            reason = response.finish_reason
        text = ''.join(parts).strip()
        # EOS is a terminator, not part of repetition statistics.
        if ids and ids[-1] in tokenizer.eos_token_ids:
            ids.pop()
        rejected = rejection(text, ids, reason)
        if not rejected and accepted is not None and text not in accepted:
            rejected = 'invalid_action'
        if text in variants and not rejected:
            rejected = 'duplicate'
        receipts.append({'seed': seed + index, 'text': text, 'tokens': len(ids),
                         'finish_reason': reason, 'rejection': rejected,
                         'full_prompt_tokens': len(attempt_prompt), 'cached_prefix_tokens': reused,
                         'seconds': time.monotonic() - started,
                         **({'messages': attempt_messages} if diverse else {})})
        if ids:
            receipts[-1].update(prompt_tokens=getattr(response, 'prompt_tokens', None),
                               prompt_tps=getattr(response, 'prompt_tps', None),
                               generation_tps=getattr(response, 'generation_tps', None))
        if not rejected:
            variants.append(text)
            if on_variant:
                on_variant(list(variants), list(receipts))
        if len(variants) == count:
            break
    return variants, receipts

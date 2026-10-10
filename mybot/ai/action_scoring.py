"""Choose a trained action from two exact final-answer completions."""

import math

from mybot.ai.local_format import prompt_text


def score_actions(model, tokenizer, messages):
    import mlx.core as mx
    prefix = prompt_text(tokenizer, messages)
    before = tokenizer.encode(prefix, add_special_tokens=False)
    if not before or len(before) > 8192:
        raise ValueError('Контекст решения слишком длинный.')
    scores = {}
    for action in ('REPLY', 'WAIT'):
        full = tokenizer.encode(prefix + action, add_special_tokens=False)
        if full[:len(before)] != before or len(full) == len(before):
            raise ValueError('Токены решения не совпали с учебным префиксом.')
        target = full[len(before):]
        logits = model(mx.array([before + target[:-1]]))
        values = []
        for index, token in enumerate(target):
            row = logits[0, len(before) - 1 + index]
            value = row[token] - mx.logsumexp(row)
            mx.eval(value)
            values.append(float(value))
        score = sum(values) / len(values)
        if not math.isfinite(score):
            raise ValueError('Локальная модель вернула некорректную вероятность действия.')
        scores[action] = score
    return max(scores, key=scores.get), scores

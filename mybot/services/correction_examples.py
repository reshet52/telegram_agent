"""Small, local retrieval of explicitly confirmed reply examples."""

import json
import re

from mybot.storage.reply_journal import finalized_correction_candidates


MAX_EXAMPLES = 3
MAX_TOTAL_CHARS = 1800
MAX_FIELD_CHARS = 220
COMMON_TERMS = {
    'about', 'and', 'are', 'for', 'from', 'have', 'how', 'the', 'this',
    'was', 'were', 'what', 'where', 'with', 'you', 'your',
    'его', 'для', 'как', 'меня', 'мне', 'мои', 'мой', 'моя', 'она',
    'они', 'про', 'так', 'тебя', 'тебе', 'твой', 'что', 'это',
}


def _terms(text):
    return {word for word in re.findall(r"\w+", text.casefold())
            if len(word) >= 3 and word not in COMMON_TERMS}


def relevant_correction_examples(workspace, current_incoming,
                                 limit=MAX_EXAMPLES, max_chars=MAX_TOTAL_CHARS):
    """Rank same-dialog examples by trigger overlap; never guess from pending data."""
    query = _terms(current_incoming)
    if not query or limit <= 0 or max_chars <= 0:
        return []
    ranked = []
    for index, item in enumerate(finalized_correction_candidates(workspace)):
        terms = _terms(item['trigger_text'])
        shared = query & terms
        if not shared:
            continue
        score = len(shared) / len(query | terms)
        ranked.append((score, -index, item))
    ranked.sort(reverse=True)
    selected = []
    used = 0
    for _, _, item in ranked:
        example = {
            'generation_id': item['generation_id'],
            'result': item['result'],
            'incoming': item['trigger_text'][:MAX_FIELD_CHARS],
            'ai_variant': item['original_text'][:MAX_FIELD_CHARS],
            'actual_reply': item['actual_text'][:MAX_FIELD_CHARS],
        }
        size = len(json.dumps(example, ensure_ascii=False))
        if used + size > max_chars:
            continue
        selected.append(example)
        used += size
        if len(selected) == limit:
            break
    return selected


def format_correction_examples(examples):
    if not examples:
        return "Подтверждённых примеров для этой ситуации пока нет."
    return json.dumps(examples, ensure_ascii=False, indent=2)

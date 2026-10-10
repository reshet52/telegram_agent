"""Small local-only facts/settings block, without API prompts or embeddings calls."""

import json
import re

from mybot.storage.atomic import read_json
from mybot.storage.dialog_goal import load_goal
from mybot.storage.dialog_behavior import load_behavior


def build_local_context(messages, instruction, memory, workspace):
    if len(instruction) > 2500:
        raise ValueError('Разовое указание слишком длинное для локальной модели (2500 символов).')
    # Owner constraints remain complete; only background material is bounded.
    facts = {'goal': load_goal(workspace), 'instruction': instruction,
             'behavior': load_behavior(workspace)}
    profiles = {key: json.dumps(memory.get(key, {}), ensure_ascii=False)[:1000]
                for key in ('user_profile', 'person_profile')}
    facts['profiles'] = profiles
    if workspace:
        records = read_json(workspace.agent_memory, [])
        if isinstance(records, dict):
            records = [r for value in records.values() if isinstance(value, list) for r in value]
        if not isinstance(records, list):
            records = []
        # Match cached evidence locally; never pay for embeddings on a local reply.
        words = set(re.findall(r'\w{3,}', '\n'.join(m.get('text') or '' for m in messages[-5:]).lower()))
        scored = [(len(words & set(re.findall(r'\w{3,}', str(r.get('memory', '')).lower()))), r)
                  for r in records if isinstance(r, dict)]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        facts['relevant_memory'] = [str(r.get('memory', ''))[:500] for score, r in scored[:3] if score]
    return json.dumps(facts, ensure_ascii=False)

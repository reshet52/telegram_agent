"""The same final-answer prefix and completion boundary for MLX SFT and inference."""

from collections import Counter
import re

FORMAT_VERSION = 'qwen-final-v1'
EMPTY_THINK = '<think>\n\n</think>\n\n'
END = '<|im_end|>'


def prompt_text(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not rendered.endswith('<|im_start|>assistant\n'):
        raise ValueError('Локальный шаблон не совместим с форматом окончательного ответа.')
    return rendered + EMPTY_THINK


def prompt_tokens(tokenizer, messages):
    return tokenizer.encode(prompt_text(tokenizer, messages), add_special_tokens=False)


def training_tokens(tokenizer, row):
    if len(row['completion']) != 1 or row['completion'][0]['role'] != 'assistant':
        raise ValueError('Нужен один целевой ответ владельца.')
    target = row['completion'][0]['content']
    if not target.strip() or re.search(r'</?think>|<\|[^>]+\|>', target, re.I):
        raise ValueError('Целевой ответ пуст или содержит служебные маркеры.')
    prefix = prompt_text(tokenizer, row['prompt'])
    before = tokenizer.encode(prefix, add_special_tokens=False)
    tokens = tokenizer.encode(prefix + target + END, add_special_tokens=False)
    if tokens[:len(before)] != before or len(tokens) <= len(before):
        raise ValueError('Граница учебного ответа не совпадает с префиксом генерации.')
    return tokens, len(before)


def rejection(text, token_ids, finish_reason):
    if not text.strip():
        return 'empty'
    if re.search(r'</?think\b|<\|[^>]+\|>', text, re.I):
        return 'service_tokens'
    if finish_reason != 'stop':
        return 'token_limit' if finish_reason == 'length' else 'unfinished'
    for width in range(1, 5):
        for start in range(len(token_ids) - width * 6 + 1):
            if token_ids[start:start + width] * 6 == token_ids[start:start + width * 6]:
                return 'repetition'
    if len(token_ids) >= 32 and max(Counter(token_ids).values(), default=0) > len(token_ids) / 2:
        return 'dominant_token'
    return None


class FinalAnswerDataset:
    def __init__(self, rows, tokenizer):
        self.items = [training_tokens(tokenizer, row) for row in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

"""Balanced chronological sampling, with redistribution of empty quotas."""

import random

PERIODS = ('Начало', 'Середина', 'Недавняя часть')


def boundaries(ids):
    ids = sorted(ids)
    if not ids:
        raise ValueError('Нет подходящих отрывков.')
    return [ids[len(ids) // 3], ids[2 * len(ids) // 3]]


def period(key, cuts):
    return int(key >= cuts[0]) + int(key >= cuts[1])


def balanced(items, limit, *, extra=2, seed=42):
    rng = random.Random(seed)
    groups = [[item for item in items if item['period'] == i] for i in range(3)]
    for group in groups:
        rng.shuffle(group)
    quotas = [limit // 3] * 3
    for i in range(limit % 3):
        quotas[(extra + i) % 3] += 1
    selected = []
    for group, quota in zip(groups, quotas):
        count = min(len(group), quota)
        selected.extend(group[:count])
        del group[:count]
    while len(selected) < limit and any(groups):
        for group in groups:
            if group and len(selected) < limit:
                selected.append(group.pop())
    rng.shuffle(selected)
    return selected

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(relevance: Sequence[int | float | bool], k: int) -> float:
    values = list(relevance)[:k]
    if not values or k <= 0:
        return 0.0
    return sum(1 for value in values if value) / len(values)


def recall_at_k(relevance: Sequence[int | float | bool], total_relevant: int, k: int) -> float:
    if total_relevant <= 0 or k <= 0:
        return 0.0
    values = list(relevance)[:k]
    found = sum(1 for value in values if value)
    return found / total_relevant


def ndcg_at_k(relevance: Sequence[int | float], k: int) -> float:
    values = list(relevance)[:k]
    if not values or k <= 0:
        return 0.0

    dcg = _dcg(values)
    ideal = _dcg(sorted(values, reverse=True))
    if ideal == 0:
        return 0.0
    return dcg / ideal


def _dcg(values: Sequence[int | float]) -> float:
    return sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))


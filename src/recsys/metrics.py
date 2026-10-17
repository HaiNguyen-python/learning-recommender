"""Offline ranking metrics. Each takes one ranked list and the set of relevant lessons.

Relevant = lessons the learner *completed* in the test window and had not started before.
Completion, not clicks, is the outcome a learning product cares about.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Share of the relevant lessons that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Binary-gain NDCG: hits near the top count more; 1.0 = all possible hits at the top."""
    rel = set(relevant)
    dcg = sum(1.0 / np.log2(pos + 2) for pos, item in enumerate(ranked[:k]) if item in rel)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(rel), k)))
    return float(dcg / ideal) if ideal > 0 else 0.0


def average_precision_at_k(ranked: Sequence[int], relevant: Collection[int], k: int) -> float:
    """AP@k = mean of precision@i over the positions i of the hits, divided by min(|rel|, k)."""
    rel = set(relevant)
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for pos, item in enumerate(ranked[:k]):
        if item in rel:
            hits += 1
            total += hits / (pos + 1)
    return total / min(len(rel), k)


def catalogue_coverage(lists: Sequence[Sequence[int]], n_items: int) -> float:
    """Share of the catalogue that appears in at least one recommendation list."""
    shown = {i for lst in lists for i in lst}
    return len(shown) / n_items


def novelty(lists: Sequence[Sequence[int]], novelty_bits: np.ndarray) -> float:
    """Mean self-information -log2(share of learners who started the lesson) of recommended
    lessons. Higher = less obvious recommendations."""
    values = [novelty_bits[i] for lst in lists for i in lst]
    return float(np.mean(values)) if values else 0.0

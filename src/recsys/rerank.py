"""Learning-aware re-ranking: relevance from a model, then rules a teacher would apply.

A collaborative model answers "what will this learner open?". A learning product also needs
"can they finish it, and will it help?". So the top candidates from the model are re-scored:

    final_i = rel_i - level_penalty * max(0, level_i - est_level - tolerance)
                    - easy_penalty  * max(0, est_level - level_i - easy_margin)

- rel_i is the model score min-max scaled to [0, 1] over the candidates.
- est_level comes from `features.estimate_level` (the same function as the feature table).
- Lessons with an unmet prerequisite are filtered out. They are only used to fill the list if
  fewer than k eligible lessons remain, and then they go last.
- Optional MMR (maximal marginal relevance, Carbonell and Goldstein 1998) trades a little
  relevance for topic variety: pick the lesson maximising
  lambda * final_i - (1 - lambda) * max_{j already picked} same_topic(i, j).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .domain import Catalog, LearnerHistory


@dataclass(frozen=True)
class RerankConfig:
    n_candidates: int = 60
    level_tolerance: float = 0.5
    level_penalty: float = 1.0
    easy_margin: float = 1.5
    easy_penalty: float = 0.3
    mmr_lambda: float | None = None  # None = no diversity step


def prereqs_met(lesson: int, history: LearnerHistory, catalog: Catalog, est_level: float) -> bool:
    """A prerequisite counts as met if it was completed, or if the learner's estimated level is
    a full level above it (placement: an advanced learner need not redo beginner lessons)."""
    return all(
        p in history.completed or est_level >= catalog.level[p] + 1 for p in catalog.prereqs[lesson]
    )


def learning_scores(
    rel: np.ndarray, levels: np.ndarray, est_level: float, cfg: RerankConfig
) -> np.ndarray:
    """Apply the level penalties to relevance scores already scaled to [0, 1]."""
    too_hard = np.maximum(0.0, levels - est_level - cfg.level_tolerance)
    too_easy = np.maximum(0.0, est_level - levels - cfg.easy_margin)
    return rel - cfg.level_penalty * too_hard - cfg.easy_penalty * too_easy


def mmr_order(scores: np.ndarray, topics: np.ndarray, k: int, lam: float) -> list[int]:
    """Greedy MMR over positions 0..n-1 with "same topic" as the similarity."""
    chosen: list[int] = []
    left = list(range(len(scores)))
    while left and len(chosen) < k:
        picked_topics = {int(topics[c]) for c in chosen}
        values = [lam * scores[i] - (1 - lam) * (int(topics[i]) in picked_topics) for i in left]
        best = left[int(np.argmax(values))]
        chosen.append(best)
        left.remove(best)
    return chosen


def rerank(
    base: np.ndarray,
    history: LearnerHistory,
    catalog: Catalog,
    est_level: float,
    k: int,
    cfg: RerankConfig,
) -> list[int]:
    """Top-k lessons after the learning rules. Already started lessons are never returned."""
    s = base.astype(float).copy()
    s[list(history.started)] = -np.inf
    order = np.lexsort((np.arange(len(s)), -s))
    cand = np.array([i for i in order[: cfg.n_candidates] if np.isfinite(s[i])], dtype=int)
    if len(cand) == 0:
        return []
    raw = s[cand]
    span = raw.max() - raw.min()
    rel = (raw - raw.min()) / span if span > 0 else np.ones(len(cand))
    final = learning_scores(rel, catalog.level[cand], est_level, cfg)
    ok = np.array([prereqs_met(int(i), history, catalog, est_level) for i in cand])
    eligible, blocked = cand[ok], cand[~ok]
    f_ok = final[ok]
    if cfg.mmr_lambda is None:
        picks = list(np.lexsort((eligible, -f_ok))[:k])
    else:
        picks = mmr_order(f_ok, catalog.topic[eligible], k, cfg.mmr_lambda)
    result = [int(eligible[p]) for p in picks]
    if len(result) < k:
        fill = blocked[np.lexsort((blocked, -final[~ok]))]
        result += [int(i) for i in fill[: k - len(result)]]
    return result

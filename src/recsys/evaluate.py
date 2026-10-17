"""Offline evaluation with a time-based split.

1. Split the log at a cutoff time T (`data.time_split`).
2. Fit every model on events before T only.
3. For each learner who completes something new after T, recommend k lessons from their
   history before T, and compare with what they completed after T.
4. Report recall@k, NDCG@k, MAP@k, catalogue coverage and novelty, for all learners and
   separately for warm and cold-start learners (at most 3 lessons started before T).

When the data comes from the simulator, an extra column `too_hard` uses the ground truth: the
share of recommended lessons more than one level above the learner's true skill at T (an unmet
prerequisite counts as one more level). Real logs cannot measure this; that is the point of
having a simulator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .als import ImplicitALS
from .baselines import ContentPopularity, ItemKNN, MostPopular, NextInSequence
from .data import heldout_targets, time_split
from .domain import Catalog, LearnerHistory, histories_from_events
from .features import item_features
from .metrics import average_precision_at_k, catalogue_coverage, ndcg_at_k, novelty, recall_at_k
from .policy import COLD_START_MAX, LearningRecommender, PlainRanker, Ranker
from .rerank import RerankConfig

GapFn = Callable[[str, LearnerHistory, int], float]
HYBRID = "ALS + re-rank + fallback"
FALLBACK = "Content/popularity + re-rank"
SEGMENTS = ("all", "warm", "cold start")


@dataclass(frozen=True)
class ALSConfig:
    # Chosen once on a separate tuning world (seed 7), never on the reported test window.
    factors: int = 16
    reg: float = 30.0
    alpha: float = 2.0
    iterations: int = 12

    def build(self) -> ImplicitALS:
        return ImplicitALS(self.factors, self.reg, self.alpha, self.iterations)


@dataclass(frozen=True)
class OfflineConfig:
    k: int = 10
    als: ALSConfig = field(default_factory=ALSConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    mmr_lambda: float = 0.7
    cold_start_max: int = COLD_START_MAX
    include_deep: bool = False  # add the two-tower model (needs the [deep] extra)


@dataclass
class OfflineResult:
    table: pd.DataFrame  # index (model, segment); metric columns + n_learners
    cutoff: float
    n_train_events: int
    als: ImplicitALS
    recommendations: dict[str, dict[str, list[int]]]  # model -> learner -> list


def build_rankers(
    catalog: Catalog, histories: Mapping[str, LearnerHistory], cfg: OfflineConfig
) -> tuple[dict[str, Ranker], ImplicitALS]:
    """Fit every model on the given (training) histories and wrap them as rankers."""
    n = len(catalog)
    als = cfg.als.build().fit(histories, n)
    content = ContentPopularity(catalog).fit(histories, n)
    mmr = replace(cfg.rerank, mmr_lambda=cfg.mmr_lambda)
    rankers: dict[str, Ranker] = {
        "Most popular": PlainRanker(MostPopular().fit(histories, n), catalog),
        "Next in sequence": PlainRanker(NextInSequence(catalog).fit(histories, n), catalog),
        "Item kNN (cosine)": PlainRanker(ItemKNN().fit(histories, n), catalog),
        "ALS (implicit)": PlainRanker(als, catalog),
        "ALS + re-rank": LearningRecommender(catalog, als, cfg.rerank, name="ALS + re-rank"),
        "ALS + re-rank + MMR": LearningRecommender(catalog, als, mmr, name="ALS + re-rank + MMR"),
        FALLBACK: LearningRecommender(catalog, content, cfg.rerank, name=FALLBACK),
        HYBRID: LearningRecommender(catalog, als, cfg.rerank, content, cfg.cold_start_max, HYBRID),
    }
    if cfg.include_deep:
        from .two_tower import TwoTower

        rankers[TwoTower.name] = PlainRanker(TwoTower(catalog).fit(histories, n), catalog)
    return rankers, als


def score_lists(
    lists: Mapping[str, list[int]],
    targets: Mapping[str, set[int]],
    learners: list[str],
    k: int,
    n_items: int,
    novelty_bits: np.ndarray,
    too_hard: Mapping[str, float] | None,
) -> dict[str, float]:
    recs = [lists[u] for u in learners]
    pairs = [(lists[u], targets[u]) for u in learners]
    out = {
        f"recall@{k}": float(np.mean([recall_at_k(r, t, k) for r, t in pairs])),
        f"ndcg@{k}": float(np.mean([ndcg_at_k(r, t, k) for r, t in pairs])),
        f"map@{k}": float(np.mean([average_precision_at_k(r, t, k) for r, t in pairs])),
        "coverage": catalogue_coverage(recs, n_items),
        "novelty": novelty(recs, novelty_bits),
        "n_learners": float(len(learners)),
    }
    if too_hard is not None:
        out["too_hard"] = float(np.mean([too_hard[u] for u in learners]))
    return out


def run_offline(
    events: pd.DataFrame,
    catalog: Catalog,
    cutoff: float,
    cfg: OfflineConfig | None = None,
    gap_fn: GapFn | None = None,
) -> OfflineResult:
    cfg = cfg or OfflineConfig()
    split = time_split(events, cutoff)
    histories = histories_from_events(split.train, catalog)
    targets = heldout_targets(split.test, histories, catalog)
    learners = sorted(targets)
    hist_of = {u: histories.get(u, LearnerHistory()) for u in learners}
    cold = [u for u in learners if hist_of[u].n_started <= cfg.cold_start_max]
    warm = [u for u in learners if hist_of[u].n_started > cfg.cold_start_max]
    bits = item_features(split.train, catalog, cutoff)["novelty_bits"].to_numpy()
    rankers, als = build_rankers(catalog, histories, cfg)
    rows, all_recs = [], {}
    for name, ranker in rankers.items():
        recs = {u: ranker.recommend(hist_of[u], cfg.k).items for u in learners}
        all_recs[name] = recs
        hard = None
        if gap_fn is not None:
            hard = {
                u: float(np.mean([gap_fn(u, hist_of[u], i) > 1.0 for i in recs[u]]))
                for u in learners
            }
        for segment, group in zip(SEGMENTS, (learners, warm, cold), strict=True):
            if group:
                m = score_lists(recs, targets, group, cfg.k, len(catalog), bits, hard)
                rows.append({"model": name, "segment": segment, **m})
    table = pd.DataFrame(rows).set_index(["model", "segment"])
    return OfflineResult(table, cutoff, len(split.train), als, all_recs)

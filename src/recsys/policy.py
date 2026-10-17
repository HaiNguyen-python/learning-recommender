"""Recommendation policies: a scorer, optionally re-ranked, optionally with a cold-start fallback.

`LearningRecommender` is the deployed policy ("B" in the A/B test, and what the API serves):

1. Estimate the learner's level from history (`features.estimate_level`).
2. If the learner has started at most `cold_start_max` lessons, score with the content +
   popularity fallback; otherwise with ALS (fold-in from the current history).
3. Re-rank the top candidates with the learning rules (level, prerequisites, optional MMR).

The bundle is saved as JSON + NPZ (no pickle), so it is readable in a code review and safe to
load.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .als import ImplicitALS
from .baselines import ContentPopularity, Scorer, top_k
from .domain import Catalog, LearnerHistory
from .features import estimate_level
from .rerank import RerankConfig, rerank

COLD_START_MAX = 3


@dataclass(frozen=True)
class Recommendation:
    items: list[int]
    strategy: str
    est_level: float


class Ranker(Protocol):
    name: str

    def recommend(self, history: LearnerHistory, k: int) -> Recommendation: ...


class PlainRanker:
    """A scorer used as is: the k best unseen lessons."""

    def __init__(self, scorer: Scorer, catalog: Catalog) -> None:
        self.scorer, self.catalog, self.name = scorer, catalog, scorer.name

    def recommend(self, history: LearnerHistory, k: int) -> Recommendation:
        items = top_k(self.scorer.score(history), history.started, k)
        return Recommendation(items, self.name, estimate_level(history, self.catalog))


class LearningRecommender:
    def __init__(
        self,
        catalog: Catalog,
        model: Scorer,
        rerank_cfg: RerankConfig | None = None,
        fallback: Scorer | None = None,
        cold_start_max: int = COLD_START_MAX,
        name: str = "ALS + re-rank + fallback",
    ) -> None:
        self.catalog = catalog
        self.model = model
        self.rerank_cfg = rerank_cfg
        self.fallback = fallback
        self.cold_start_max = cold_start_max
        self.name = name

    def recommend(self, history: LearnerHistory, k: int) -> Recommendation:
        est = estimate_level(history, self.catalog)
        cold = self.fallback is not None and history.n_started <= self.cold_start_max
        scorer = self.fallback if cold and self.fallback is not None else self.model
        base = scorer.score(history)
        if self.rerank_cfg is None:
            items = top_k(base, history.started, k)
        else:
            items = rerank(base, history, self.catalog, est, k, self.rerank_cfg)
        strategy = ("fallback: " if cold else "") + scorer.name
        return Recommendation(items, strategy, est)

    def save(self, directory: Path) -> None:
        """Only the deployed shape is saved: ALS + content/popularity fallback + re-rank rules."""
        if not isinstance(self.model, ImplicitALS) or not isinstance(
            self.fallback, ContentPopularity
        ):
            raise TypeError("only an ALS model with the content/popularity fallback is saved")
        directory.mkdir(parents=True, exist_ok=True)
        self.catalog.save(directory / "catalog.json")
        np.savez_compressed(
            directory / "model.npz",
            item_factors=self.model.item_factors,
            popularity=self.fallback.popularity,
        )
        config = {
            "als": {
                "factors": self.model.factors,
                "reg": self.model.reg,
                "alpha": self.model.alpha,
                "iterations": self.model.iterations,
            },
            "rerank": asdict(self.rerank_cfg) if self.rerank_cfg else None,
            "cold_start_max": self.cold_start_max,
            "topic_weight": self.fallback.topic_weight,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory: Path) -> LearningRecommender:
        catalog = Catalog.load(directory / "catalog.json")
        config = json.loads((directory / "config.json").read_text())
        arrays = np.load(directory / "model.npz")
        als = ImplicitALS(**config["als"])
        als.item_factors = arrays["item_factors"]
        fallback = ContentPopularity(catalog, config["topic_weight"])
        fallback.popularity = arrays["popularity"]
        rr = config["rerank"]
        return cls(
            catalog,
            als,
            RerankConfig(**rr) if rr else None,
            fallback,
            config["cold_start_max"],
        )

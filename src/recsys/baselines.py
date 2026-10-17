"""Scorers: each one turns a learner's history into a score for every lesson.

All scorers share one tiny interface (`fit`, `score`), so the evaluation, the re-ranker, the
simulated A/B test and the API can treat them the same way. Higher score = recommend sooner.
Lessons the learner already started are removed later, in `top_k`, not here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

import numpy as np

from .als import interaction_matrix
from .domain import Catalog, LearnerHistory


class Scorer(Protocol):
    name: str

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> Scorer: ...

    def score(self, history: LearnerHistory) -> np.ndarray: ...


def top_k(scores: np.ndarray, exclude: Iterable[int], k: int) -> list[int]:
    """The k best lessons not in `exclude`; ties broken by lesson index so results are stable."""
    s = scores.astype(float).copy()
    s[list(exclude)] = -np.inf
    order = np.lexsort((np.arange(len(s)), -s))
    return [int(i) for i in order[:k] if np.isfinite(s[i])]


def learner_counts(histories: Mapping[str, LearnerHistory], n_items: int) -> np.ndarray:
    """How many learners started each lesson (the popularity used everywhere)."""
    counts = np.zeros(n_items)
    for h in histories.values():
        counts[list(h.started)] += 1
    return counts


class MostPopular:
    """Same list for everyone: lessons started by the most learners."""

    name = "Most popular"

    def __init__(self) -> None:
        self.popularity = np.zeros(0)

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> MostPopular:
        self.popularity = learner_counts(histories, n_items)
        return self

    def score(self, history: LearnerHistory) -> np.ndarray:
        return self.popularity


class NextInSequence:
    """Continue the topic tracks the learner is on: the earliest lessons not yet started in the
    most recent topic, then in the topic before that, and so on. Popularity fills the rest.

    This is what many course platforms do by default ("continue where you left off").
    """

    name = "Next in sequence (same topic)"

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.popularity = np.zeros(0)

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> NextInSequence:
        self.popularity = learner_counts(histories, n_items)
        return self

    def score(self, history: LearnerHistory) -> np.ndarray:
        cat = self.catalog
        scores = 0.01 * self.popularity / (float(np.max(self.popularity)) + 1.0)
        recent_topics = list(dict.fromkeys(cat.topic[i] for i in reversed(history.lessons)))
        for rank, topic in enumerate(recent_topics):
            # Topic rank dominates; inside a topic, earlier track positions come first.
            in_topic = cat.topic == topic
            track = 1.0 + 1.0 / (1 + rank) - 0.001 * cat.seq
            scores = np.where(in_topic, track, scores)
        return scores


class ItemKNN:
    """Item-item cosine similarity on the binary learner x lesson matrix (top neighbours only).

    score(i) = sum over started lessons j of sim(i, j). Keeping only the strongest neighbours of
    each lesson removes the long tail of weak, noisy similarities.
    """

    name = "Item kNN (cosine)"

    def __init__(self, neighbours: int = 50) -> None:
        self.neighbours = neighbours
        self.similarity = np.zeros((0, 0))

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> ItemKNN:
        x = interaction_matrix(list(histories.values()), n_items)
        x.data[:] = 1.0
        co = (x.T @ x).toarray()
        norms = np.sqrt(np.diag(co))
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.nan_to_num(co / np.outer(norms, norms))
        np.fill_diagonal(sim, 0.0)
        if self.neighbours < n_items:
            # Zero everything below the k-th largest value in each column.
            kth = np.partition(sim, -self.neighbours, axis=0)[-self.neighbours]
            sim = np.where(sim >= kth[None, :], sim, 0.0)
        self.similarity = sim
        return self

    def score(self, history: LearnerHistory) -> np.ndarray:
        if not history.started:
            return np.zeros(self.similarity.shape[0])
        return self.similarity[:, list(history.started)].sum(axis=1)


class ContentPopularity:
    """Cold-start fallback: popularity, boosted inside the topics the learner has touched.

    With zero history this is plain popularity (the re-ranker then keeps it near beginner
    level). With one to three lessons, the topics are the only reliable signal: too little for a
    collaborative model, enough to say "more grammar, please".
    """

    name = "Content + popularity fallback"

    def __init__(self, catalog: Catalog, topic_weight: float = 2.0) -> None:
        self.catalog = catalog
        self.topic_weight = topic_weight
        self.popularity = np.zeros(0)

    def fit(self, histories: Mapping[str, LearnerHistory], n_items: int) -> ContentPopularity:
        self.popularity = learner_counts(histories, n_items)
        return self

    def score(self, history: LearnerHistory) -> np.ndarray:
        pop = np.log1p(self.popularity)
        pop = pop / (pop.max() + 1e-9)
        if not history.started:
            return pop
        counts = np.bincount(
            self.catalog.topic[list(history.started)], minlength=len(self.catalog.topics)
        )
        share = counts / counts.sum()
        return pop + self.topic_weight * share[self.catalog.topic]

"""Learner and item features, computed by the same functions for training and for serving.

`learner_features(history, catalog, as_of)` is the only place learner features are defined.
The batch path (`build_feature_table`) calls it for every learner in the training log; the
serving path (`serving_learner_features`) calls it for one learner looked up in the
`EventStore`. `parity_check` compares the two and `recsys train` refuses to save artefacts if
they differ. That catches the classic training/serving skew bugs: a different time filter, a
different sort order, or an event that one path forgets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .domain import Catalog, EventStore, LearnerHistory, histories_from_events

# Level estimate: a shrunk average. Each completed lesson says "can do at least level L" (vote
# L + 0.5); each abandoned one says "probably above you" (vote L - 1). Two pseudo-votes at the
# prior keep new learners near beginner level instead of jumping on one lucky completion.
PRIOR_LEVEL = 1.5
PRIOR_WEIGHT = 2.0
COMPLETE_VOTE = 0.5
ABANDON_VOTE = -1.0
NO_HISTORY = -1.0  # days_since_last when the learner has no events yet

LEARNER_FEATURES = [
    "n_started",
    "n_completed",
    "n_abandoned",
    "completion_rate",
    "est_level",
    "days_since_last",
    "active_days",
    "top_topic_share",
    "n_topics",
]


def estimate_level(history: LearnerHistory, catalog: Catalog) -> float:
    """Shrunk average of level votes; see the constants above for the reasoning."""
    total, weight = PRIOR_LEVEL * PRIOR_WEIGHT, PRIOR_WEIGHT
    for lesson in history.completed:
        total += catalog.level[lesson] + COMPLETE_VOTE
        weight += 1.0
    for lesson in history.abandoned - history.completed:
        total += catalog.level[lesson] + ABANDON_VOTE
        weight += 1.0
    return float(total / weight)


def learner_features(history: LearnerHistory, catalog: Catalog, as_of: float) -> dict[str, float]:
    """Features of one learner from events strictly before `as_of`."""
    times = [t for t in history.times if t < as_of]
    if len(times) != len(history.times):
        raise ValueError("history contains events at or after as_of")
    n_started = history.n_started
    n_completed = len(history.completed)
    topics = np.bincount(catalog.topic[list(history.started)], minlength=len(catalog.topics))
    return {
        "n_started": float(n_started),
        "n_completed": float(n_completed),
        "n_abandoned": float(len(history.abandoned - history.completed)),
        # Laplace smoothing: a learner with 1 of 1 completed is not "100% reliable".
        "completion_rate": (n_completed + 1.0) / (n_started + 2.0),
        "est_level": estimate_level(history, catalog),
        "days_since_last": float(as_of - times[-1]) if times else NO_HISTORY,
        "active_days": float(len({int(t) for t in times})),
        "top_topic_share": float(np.max(topics) / n_started) if n_started else 0.0,
        "n_topics": float((topics > 0).sum()),
    }


def item_features(events: pd.DataFrame, catalog: Catalog, as_of: float) -> pd.DataFrame:
    """Lesson features from events before `as_of` (popularity, completion rate, novelty)."""
    df = events[events["t"] < as_of]
    idx = catalog.index()
    lesson = df["lesson_id"].map(idx).to_numpy()
    n = len(catalog)
    views = df["event"].to_numpy() == "view"
    starters = df.loc[views].drop_duplicates(["learner_id", "lesson_id"])
    starts = np.bincount(starters["lesson_id"].map(idx).to_numpy(), minlength=n).astype(float)
    completes = np.bincount(lesson[df["event"].to_numpy() == "complete"], minlength=n)
    n_learners = max(df["learner_id"].nunique(), 1)
    share = np.maximum(starts, 1.0) / n_learners
    return pd.DataFrame(
        {
            "lesson_id": catalog.lesson_ids,
            "topic": catalog.topic,
            "level": catalog.level,
            "n_prereqs": [len(p) for p in catalog.prereqs],
            "starts": starts,
            # Beta(2, 2)-style smoothing toward 50% for rarely started lessons.
            "completion_rate": (completes + 2.0) / (starts + 4.0),
            # Self-information of a lesson: rare lessons are "novel" (Zhou et al. 2010).
            "novelty_bits": -np.log2(np.minimum(share, 1.0)),
        }
    ).set_index("lesson_id")


def learner_table(
    histories: dict[str, LearnerHistory], catalog: Catalog, as_of: float
) -> pd.DataFrame:
    rows = {lid: learner_features(h, catalog, as_of) for lid, h in histories.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=LEARNER_FEATURES).rename_axis(
        "learner_id"
    )


def build_feature_table(
    events: pd.DataFrame, catalog: Catalog, as_of: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batch (training) path: learner and item feature tables as of `as_of`."""
    histories = histories_from_events(events, catalog, as_of)
    return learner_table(histories, catalog, as_of), item_features(events, catalog, as_of)


def serving_learner_features(store: EventStore, learner_id: str, as_of: float) -> dict[str, float]:
    """Online (serving) path: one learner, looked up in the event store."""
    return learner_features(store.history(learner_id, as_of), store.catalog, as_of)


def parity_check(
    events: pd.DataFrame, catalog: Catalog, as_of: float, learner_ids: list[str] | None = None
) -> float:
    """Largest absolute difference between batch and serving features (0.0 means parity)."""
    batch, _ = build_feature_table(events, catalog, as_of)
    store = EventStore(events, catalog)
    ids = learner_ids if learner_ids is not None else list(batch.index)
    worst = 0.0
    for lid in ids:
        online = serving_learner_features(store, lid, as_of)
        offline = batch.loc[lid].to_numpy(dtype=float)
        diff = np.abs(np.array([online[c] for c in LEARNER_FEATURES]) - offline)
        worst = max(worst, float(diff.max()))
    return worst

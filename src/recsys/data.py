"""Loading event logs and splitting them by time.

The canonical event table has four columns: `learner_id, lesson_id, t, event`, where `t` is in
days and `event` is `view`, `complete` or `abandon`. The simulator writes this format, and the
loaders below map public implicit-feedback data into it.

Real datasets (not shipped, check their terms before downloading):

- **Canonical CSV** (what `recsys generate` writes): `load_events_csv`.
- **Generic user,item,timestamp CSV** (MOOCCube user-video logs exported to CSV, or any
  clickstream): `load_implicit_csv`. Each row becomes a `view`. An optional event column can
  carry `complete`/`abandon`.
- **EdNet KT1** (Choi et al. 2020): one file per student, `u<id>.csv`, with columns
  `timestamp` (Unix ms), `solving_id`, `question_id`, `user_answer`, `elapsed_time`.
  `load_ednet_kt1` reads a folder of these. An answered question counts as a started and
  completed item (`view` + `complete`), since KT1 has no "gave up" signal.

Public logs have no lesson levels or prerequisites, so `catalog_from_events` builds a flat
catalogue (one topic, level 1, no prerequisites). The level rules then have nothing to act on
and the re-ranker reduces to the base model; only popularity, kNN and ALS are meaningful there.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .domain import EVENT_COLUMNS, EVENTS, Catalog, LearnerHistory

MS_PER_DAY = 86_400_000.0


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: float


def time_split(events: pd.DataFrame, cutoff: float) -> TimeSplit:
    """Train = everything strictly before `cutoff`; test = everything at or after it.

    One global cutoff for all learners, like deploying a model on a given day. A random row split
    would let the model see a learner's future while predicting their past.
    """
    train = events[events["t"] < cutoff].reset_index(drop=True)
    test = events[events["t"] >= cutoff].reset_index(drop=True)
    return TimeSplit(train, test, cutoff)


def heldout_targets(
    test: pd.DataFrame, train_histories: Mapping[str, LearnerHistory], catalog: Catalog
) -> dict[str, set[int]]:
    """Per learner: lessons completed in the test window that were not started before it.

    Re-finishing a lesson started in training is not a discovery the recommender made, and a
    recommender never shows already-started lessons, so those are excluded.
    """
    idx = catalog.index()
    done = test[test["event"] == "complete"]
    targets: dict[str, set[int]] = {}
    empty = LearnerHistory()
    for learner, lesson in zip(done["learner_id"], done["lesson_id"], strict=True):
        i = idx[lesson]
        if i not in train_histories.get(learner, empty).started:
            targets.setdefault(str(learner), set()).add(i)
    return targets


def _check(events: pd.DataFrame) -> pd.DataFrame:
    bad = set(events["event"].unique()) - set(EVENTS)
    if bad:
        raise ValueError(f"unknown event types: {sorted(bad)}")
    return (
        events[EVENT_COLUMNS].sort_values(["t", "learner_id"], kind="stable").reset_index(drop=True)
    )


def load_events_csv(path: Path) -> pd.DataFrame:
    """The canonical format (what `recsys generate` writes): learner_id, lesson_id, t, event."""
    raw = pd.read_csv(path, dtype={"learner_id": str, "lesson_id": str})
    return _check(raw)


def load_implicit_csv(
    path: Path,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    event_col: str | None = None,
) -> pd.DataFrame:
    """A user,item,timestamp CSV. Timestamps may be numbers (seconds) or date strings; `t` is
    days since the first event, so a cutoff can be given as "day 42"."""
    raw = pd.read_csv(path)
    ts = raw[time_col]
    if pd.api.types.is_numeric_dtype(ts):
        seconds = ts.to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(ts, utc=True)
        seconds = (parsed - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
    events = pd.DataFrame(
        {
            "learner_id": raw[user_col].astype(str),
            "lesson_id": raw[item_col].astype(str),
            "t": (seconds - seconds.min()) / 86_400.0,
            "event": raw[event_col].astype(str) if event_col else "view",
        }
    )
    return _check(events)


def load_ednet_kt1(directory: Path) -> pd.DataFrame:
    """A folder of EdNet KT1 `u<id>.csv` files -> canonical events (`view` + `complete`)."""
    frames = []
    for f in sorted(directory.glob("u*.csv")):
        raw = pd.read_csv(f, usecols=["timestamp", "question_id"])
        base = pd.DataFrame(
            {
                "learner_id": f.stem,
                "lesson_id": raw["question_id"].astype(str),
                "ms": raw["timestamp"].to_numpy(dtype=float),
            }
        )
        frames += [base.assign(event="view"), base.assign(event="complete", ms=base["ms"] + 1)]
    if not frames:
        raise FileNotFoundError(f"no u*.csv files in {directory}")
    df = pd.concat(frames, ignore_index=True)
    df["t"] = (df["ms"] - df["ms"].min()) / MS_PER_DAY
    return _check(df)


def catalog_from_events(events: pd.DataFrame) -> Catalog:
    """A flat catalogue for public data: one topic, level 1, no prerequisites."""
    ids = tuple(sorted(events["lesson_id"].astype(str).unique()))
    n = len(ids)
    return Catalog(
        topics=("unknown",),
        topic=np.zeros(n, dtype=int),
        level=np.ones(n, dtype=int),
        seq=np.arange(n),
        prereqs=tuple(() for _ in range(n)),
        ids=ids,
    )

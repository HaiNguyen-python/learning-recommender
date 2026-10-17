"""Core data types shared by the simulator, the models, the evaluation and the API.

- `Catalog`: the public lesson catalogue (topic, difficulty level, order, prerequisites).
- `LearnerHistory`: one learner's events in time order, with the running sets every model needs.
- `EventStore`: the serving-side lookup of histories by learner id.

Keeping these in one small module means the offline code, the simulator and the API all read
history the same way, which is what makes the offline/online feature parity check meaningful.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = ("view", "complete", "abandon")
# Implicit-feedback strength r_ui (Hu, Koren, Volinsky 2008). A started lesson counts 1 (the
# view); finishing it adds 2. Abandoning adds nothing: the learner still chose the lesson, so it
# remains weak positive evidence of interest, not a negative label.
EVENT_WEIGHT = {"view": 1.0, "complete": 2.0, "abandon": 0.0}
EVENT_COLUMNS = ["learner_id", "lesson_id", "t", "event"]


@dataclass(frozen=True)
class Catalog:
    """Public lesson metadata. Indices 0..n-1 are used internally; ids are for people and APIs."""

    topics: tuple[str, ...]
    topic: np.ndarray  # int topic index per lesson
    level: np.ndarray  # int difficulty 1..5
    seq: np.ndarray  # position of the lesson inside its topic track
    prereqs: tuple[tuple[int, ...], ...]
    ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.topic)

    @property
    def lesson_ids(self) -> list[str]:
        return list(self.ids)

    def index(self) -> dict[str, int]:
        return {lid: i for i, lid in enumerate(self.lesson_ids)}

    def to_frame(self) -> pd.DataFrame:
        ids = self.lesson_ids
        return pd.DataFrame(
            {
                "lesson_id": ids,
                "topic": [self.topics[t] for t in self.topic],
                "level": self.level,
                "seq": self.seq,
                "prereqs": [" ".join(ids[p] for p in ps) for ps in self.prereqs],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "topics": list(self.topics),
            "topic": self.topic.tolist(),
            "level": self.level.tolist(),
            "seq": self.seq.tolist(),
            "prereqs": [list(p) for p in self.prereqs],
            "ids": list(self.ids),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Catalog:
        prereqs = d["prereqs"]
        assert isinstance(prereqs, list)
        return cls(
            topics=tuple(d["topics"]),  # type: ignore[arg-type]
            topic=np.asarray(d["topic"], dtype=int),
            level=np.asarray(d["level"], dtype=int),
            seq=np.asarray(d["seq"], dtype=int),
            prereqs=tuple(tuple(int(x) for x in p) for p in prereqs),
            ids=tuple(str(x) for x in d["ids"]),  # type: ignore[attr-defined]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> Catalog:
        return cls.from_dict(json.loads(path.read_text()))


class LearnerHistory:
    """Events of one learner, oldest first, plus running summaries.

    `add` must be called in time order. The simulator appends live; offline code and the API
    build histories with `from_events`, which sorts first.
    """

    __slots__ = ("times", "lessons", "events", "started", "completed", "abandoned", "weight")

    def __init__(self) -> None:
        self.times: list[float] = []
        self.lessons: list[int] = []
        self.events: list[str] = []
        self.started: set[int] = set()
        self.completed: set[int] = set()
        self.abandoned: set[int] = set()
        self.weight: dict[int, float] = {}

    def add(self, t: float, lesson: int, event: str) -> None:
        if self.times and t < self.times[-1]:
            raise ValueError("events must be added in time order")
        self.times.append(t)
        self.lessons.append(lesson)
        self.events.append(event)
        self.started.add(lesson)
        if event == "complete":
            self.completed.add(lesson)
        elif event == "abandon":
            self.abandoned.add(lesson)
        self.weight[lesson] = self.weight.get(lesson, 0.0) + EVENT_WEIGHT[event]

    @classmethod
    def from_events(cls, rows: Iterable[tuple[float, int, str]]) -> LearnerHistory:
        h = cls()
        # Sort by time; at equal times a view comes before its outcome.
        for t, lesson, event in sorted(rows, key=lambda r: (r[0], EVENTS.index(r[2]))):
            h.add(float(t), int(lesson), str(event))
        return h

    @property
    def n_started(self) -> int:
        return len(self.started)

    @property
    def last_lesson(self) -> int:
        return self.lessons[-1] if self.lessons else -1


def histories_from_events(
    events: pd.DataFrame, catalog: Catalog, as_of: float | None = None
) -> dict[str, LearnerHistory]:
    """Batch path: one history per learner from a whole event table, using events before `as_of`.

    This is how the training feature table and the offline evaluation see history.
    """
    df = events if as_of is None else events[events["t"] < as_of]
    idx = catalog.index()
    lesson = df["lesson_id"].map(idx).to_numpy()
    out: dict[str, LearnerHistory] = {}
    times = df["t"].to_numpy(dtype=float)
    kinds = df["event"].to_numpy()
    for learner, rows in df.groupby("learner_id", sort=True).indices.items():
        out[str(learner)] = LearnerHistory.from_events(
            zip(times[rows], lesson[rows], kinds[rows], strict=True)
        )
    return out


class EventStore:
    """Serving path: events kept per learner, looked up one learner at a time.

    In production this would be a key-value store fed by the event stream. Here it is a dict of
    small arrays built once from the saved event log.
    """

    def __init__(self, events: pd.DataFrame, catalog: Catalog) -> None:
        self.catalog = catalog
        idx = catalog.index()
        self._rows: dict[str, list[tuple[float, int, str]]] = {}
        for learner, lesson, t, event in events[EVENT_COLUMNS].itertuples(index=False):
            self._rows.setdefault(str(learner), []).append((float(t), idx[lesson], str(event)))

    def __contains__(self, learner_id: str) -> bool:
        return learner_id in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def history(self, learner_id: str, as_of: float) -> LearnerHistory:
        """Events strictly before `as_of`; an unknown learner gets an empty history."""
        rows = self._rows.get(learner_id, [])
        return LearnerHistory.from_events(r for r in rows if r[0] < as_of)

"""`recsys train`: fit the deployed policy and save everything the API needs.

artifacts/
  catalog.json          public lesson catalogue
  model.npz             ALS lesson factors + lesson popularity (no pickle)
  config.json           ALS, re-rank and fallback settings
  events.csv.gz         the event store the API reads histories from
  learner_features.csv  feature table snapshot (batch path)
  item_features.csv
  meta.json             as_of time, sizes, ALS loss per iteration, parity result

Training stops before saving if the batch and serving feature paths disagree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .baselines import ContentPopularity
from .domain import Catalog, EventStore, histories_from_events
from .evaluate import HYBRID, OfflineConfig
from .features import build_feature_table, parity_check
from .policy import LearningRecommender


@dataclass
class Served:
    recommender: LearningRecommender
    store: EventStore
    as_of: float


def train_artifacts(
    events: pd.DataFrame, catalog: Catalog, as_of: float, out: Path, cfg: OfflineConfig
) -> dict[str, object]:
    histories = histories_from_events(events, catalog, as_of)
    n = len(catalog)
    als = cfg.als.build().fit(histories, n)
    content = ContentPopularity(catalog).fit(histories, n)
    rec = LearningRecommender(catalog, als, cfg.rerank, content, cfg.cold_start_max, HYBRID)
    parity = parity_check(events, catalog, as_of)
    if parity > 0.0:
        raise RuntimeError(f"training/serving feature mismatch: max difference {parity}")
    learners, items = build_feature_table(events, catalog, as_of)
    rec.save(out)
    events[events["t"] < as_of].to_csv(out / "events.csv.gz", index=False)
    learners.to_csv(out / "learner_features.csv")
    items.to_csv(out / "item_features.csv")
    meta: dict[str, object] = {
        "as_of": as_of,
        "n_events": int((events["t"] < as_of).sum()),
        "n_learners": len(histories),
        "n_lessons": n,
        "als_loss": [round(x, 2) for x in als.loss_history],
        "feature_parity_max_diff": parity,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_artifacts(out: Path) -> Served:
    rec = LearningRecommender.load(out)
    meta = json.loads((out / "meta.json").read_text())
    events = pd.read_csv(out / "events.csv.gz", dtype={"learner_id": str, "lesson_id": str})
    return Served(rec, EventStore(events, rec.catalog), float(meta["as_of"]))

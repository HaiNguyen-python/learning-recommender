import pytest
from conftest import CUTOFF

from recsys.domain import EventStore, LearnerHistory
from recsys.features import (
    LEARNER_FEATURES,
    NO_HISTORY,
    PRIOR_LEVEL,
    build_feature_table,
    estimate_level,
    learner_features,
    parity_check,
    serving_learner_features,
)


def test_offline_and_online_features_are_identical(events, catalog):
    assert parity_check(events, catalog, CUTOFF) == 0.0
    batch, _ = build_feature_table(events, catalog, CUTOFF)
    store = EventStore(events, catalog)
    lid = batch.index[0]
    online = serving_learner_features(store, lid, CUTOFF)
    assert [online[c] for c in LEARNER_FEATURES] == batch.loc[lid].tolist()


def test_learner_features_on_a_hand_history(catalog):
    h = LearnerHistory()
    h.add(1.2, 0, "view")
    h.add(1.21, 0, "complete")  # level 1
    h.add(3.5, 12, "view")
    h.add(3.51, 12, "abandon")  # level 3
    f = learner_features(h, catalog, as_of=5.0)
    assert f["n_started"] == 2 and f["n_completed"] == 1 and f["n_abandoned"] == 1
    assert f["completion_rate"] == pytest.approx(2 / 4)
    assert f["days_since_last"] == pytest.approx(5.0 - 3.51)
    assert f["active_days"] == 2
    # (1.5*2 + (1+0.5) + (3-1)) / 4
    assert f["est_level"] == pytest.approx((3.0 + 1.5 + 2.0) / 4)


def test_level_estimate_starts_at_the_prior_and_rejects_future_events(catalog):
    empty = LearnerHistory()
    assert estimate_level(empty, catalog) == PRIOR_LEVEL
    assert learner_features(empty, catalog, 1.0)["days_since_last"] == NO_HISTORY
    h = LearnerHistory()
    h.add(2.0, 0, "view")
    with pytest.raises(ValueError):
        learner_features(h, catalog, as_of=1.0)


def test_unknown_learner_gets_an_empty_history(events, catalog):
    store = EventStore(events, catalog)
    assert "nobody" not in store
    assert store.history("nobody", CUTOFF).n_started == 0

import numpy as np
from conftest import CUTOFF

from recsys.als import ImplicitALS
from recsys.data import heldout_targets, time_split
from recsys.domain import histories_from_events
from recsys.features import build_feature_table


def test_split_puts_every_event_on_one_side_of_the_cutoff(events):
    split = time_split(events, CUTOFF)
    assert split.train["t"].max() < CUTOFF <= split.test["t"].min()
    assert len(split.train) + len(split.test) == len(events)


def test_targets_are_new_completions_only(events, catalog, train_histories):
    split = time_split(events, CUTOFF)
    targets = heldout_targets(split.test, train_histories, catalog)
    assert targets
    for learner, lessons in targets.items():
        started_before = train_histories[learner].started if learner in train_histories else set()
        assert not lessons & started_before


def _scramble_future(events):
    """Rewrite everything after the cutoff: other lessons, other outcomes, extra rows."""
    future = events["t"] >= CUTOFF
    fake = events.copy()
    rng = np.random.default_rng(0)
    fake.loc[future, "lesson_id"] = rng.permutation(fake.loc[future, "lesson_id"].to_numpy())
    fake.loc[future, "event"] = "complete"
    return fake


def test_models_and_features_ignore_the_future(events, catalog):
    fake = _scramble_future(events)
    real_h = histories_from_events(time_split(events, CUTOFF).train, catalog)
    fake_h = histories_from_events(time_split(fake, CUTOFF).train, catalog)
    a = ImplicitALS(factors=8, iterations=3).fit(real_h, len(catalog))
    b = ImplicitALS(factors=8, iterations=3).fit(fake_h, len(catalog))
    np.testing.assert_array_equal(a.item_factors, b.item_factors)
    la, ia = build_feature_table(events, catalog, CUTOFF)
    lb, ib = build_feature_table(fake, catalog, CUTOFF)
    assert la.equals(lb) and ia.equals(ib)

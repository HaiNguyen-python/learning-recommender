import numpy as np
import pandas as pd

from recsys.domain import LearnerHistory
from recsys.simulator import (
    difficulty_gap,
    generate_history,
    initial_state,
    make_catalog,
    make_world,
    p_complete,
)


def test_same_seed_same_world_and_log():
    a, b = make_world(60, 10, seed=5), make_world(60, 10, seed=5)
    ea, _ = generate_history(a, 10, seed=9)
    eb, _ = generate_history(b, 10, seed=9)
    pd.testing.assert_frame_equal(ea, eb)
    ec, _ = generate_history(a, 10, seed=10)
    assert not ea.equals(ec)


def test_catalogue_structure():
    cat, appeal = make_catalog(seed=0)
    assert len(cat) == 300 and len(appeal) == 300
    assert len(set(cat.lesson_ids)) == 300
    for lesson, reqs in enumerate(cat.prereqs):
        assert (len(reqs) == 0) == (cat.level[lesson] == 1)
        for p in reqs:
            assert cat.topic[p] == cat.topic[lesson] and cat.level[p] == cat.level[lesson] - 1


def test_planted_difficulty_and_prerequisite_effects():
    world = make_world(5, 5, seed=0)
    cat = world.catalog
    h = LearnerHistory()
    easy = int(np.flatnonzero((cat.topic == 0) & (cat.level == 1))[0])
    hard = int(np.flatnonzero((cat.topic == 0) & (cat.level == 4))[0])
    assert p_complete(world, 0, 1.5, h, easy) > p_complete(world, 0, 1.5, h, hard) + 0.4
    # Lesson at level 2 with an unmet level-1 prerequisite feels one level harder ...
    mid = int(np.flatnonzero((cat.topic == 0) & (cat.level == 2))[0])
    assert difficulty_gap(world, 1.8, h, mid) == 2 - 1.8 + 1.0
    # ... unless the prerequisite was completed or the learner is clearly past it.
    for p in cat.prereqs[mid]:
        h.add(float(p), p, "view")
        h.add(p + 0.01, p, "complete")
    assert difficulty_gap(world, 1.8, h, mid) == 2 - 1.8
    assert difficulty_gap(world, 2.0, LearnerHistory(), mid) == 0.0


def test_history_is_time_ordered_and_late_arrivals_start_later(world, events):
    first = events.groupby("learner_id")["t"].min()
    arrival = dict(zip(world.population.learner_ids, world.population.arrival, strict=True))
    assert all(first[lid] >= arrival[lid] for lid in first.index)
    assert (world.population.arrival > 0).any()
    state = initial_state(world)
    assert all(len(h.times) == 0 for h in state.histories)

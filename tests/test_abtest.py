import numpy as np
import pandas as pd
import pytest

from recsys.abtest import analyse, cuped, ratio_effect, run_ab, welch_effect
from recsys.baselines import ContentPopularity, MostPopular
from recsys.domain import histories_from_events
from recsys.evaluate import ALSConfig
from recsys.policy import LearningRecommender, PlainRanker
from recsys.rerank import RerankConfig


def test_cuped_keeps_the_effect_and_shrinks_the_interval():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.gamma(2.0, 1.0, n)
    arm = rng.permutation(n) % 2 == 1
    y = 0.8 * x + rng.normal(0, 0.5, n) + 0.3 * arm
    raw = welch_effect(y[arm], y[~arm])
    y_adj, theta = cuped(y, x)
    adj = welch_effect(y_adj[arm], y_adj[~arm])
    assert theta == pytest.approx(0.8, abs=0.05)
    assert adj.ci_low < 0.3 < adj.ci_high
    assert (adj.ci_high - adj.ci_low) < 0.5 * (raw.ci_high - raw.ci_low)


def test_ratio_effect_matches_a_hand_example():
    e = ratio_effect(
        num_b=np.array([1.0, 1.0, 0.0, 2.0]),
        den_b=np.array([4.0, 4.0, 2.0, 2.0]),
        num_a=np.array([2.0, 2.0, 1.0, 3.0]),
        den_a=np.array([4.0, 4.0, 2.0, 6.0]),
    )
    assert e.treatment == pytest.approx(4 / 12) and e.control == pytest.approx(8 / 16)
    assert e.ci_low < e.diff < e.ci_high


def test_simulated_ab_runs_and_policy_b_lowers_abandonment(world, history, catalog):
    events, state = history
    hist = histories_from_events(events, catalog)
    n = len(catalog)
    a = PlainRanker(MostPopular().fit(hist, n), catalog)
    b = LearningRecommender(
        catalog,
        ALSConfig(iterations=4).build().fit(hist, n),
        RerankConfig(),
        ContentPopularity(catalog).fit(hist, n),
    )
    from recsys.abtest import ABConfig

    rows = run_ab(world, state, events, 28, a, b, ABConfig(n_days=10, pre_days=14, seed=1))
    assert isinstance(rows, pd.DataFrame) and len(rows) == len(world.population.learner_ids)
    assert set(rows["arm"]) == {"A", "B"}
    assert state.histories[0].times == history[1].histories[0].times  # input state untouched
    summary = analyse(rows)
    assert summary.abandonment.diff < 0
    assert 0.0 <= summary.variance_reduction < 1.0

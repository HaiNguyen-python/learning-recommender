import numpy as np

from recsys.baselines import ContentPopularity, MostPopular, NextInSequence, top_k
from recsys.domain import LearnerHistory
from recsys.evaluate import ALSConfig
from recsys.policy import LearningRecommender
from recsys.rerank import RerankConfig


def make_policy(catalog, train_histories):
    n = len(catalog)
    als = ALSConfig(iterations=4).build().fit(train_histories, n)
    fallback = ContentPopularity(catalog).fit(train_histories, n)
    return LearningRecommender(catalog, als, RerankConfig(), fallback)


def history_of(catalog, lessons):
    h = LearnerHistory()
    for k, i in enumerate(lessons):
        h.add(float(k), i, "view")
        h.add(k + 0.01, i, "complete")
    return h


def test_cold_start_switches_to_the_fallback(catalog, train_histories):
    policy = make_policy(catalog, train_histories)
    new = policy.recommend(LearnerHistory(), 10)
    assert new.strategy.startswith("fallback") and len(new.items) == 10
    # A brand-new learner is assumed a beginner: no lesson above level 2.
    assert catalog.level[new.items].max() <= 2
    three = policy.recommend(history_of(catalog, [0, 1, 2]), 10)
    assert three.strategy.startswith("fallback")
    four = policy.recommend(history_of(catalog, [0, 1, 2, 3]), 10)
    assert four.strategy == "ALS (implicit)"


def test_fallback_follows_the_topics_a_new_learner_touched(catalog, train_histories):
    policy = make_policy(catalog, train_histories)
    travel = np.flatnonzero(catalog.topic == catalog.topics.index("travel"))[:2]
    recs = policy.recommend(history_of(catalog, travel), 4).items
    assert all(catalog.topic[i] == catalog.topic[travel[0]] for i in recs)


def test_saved_bundle_gives_identical_recommendations(tmp_path, catalog, train_histories):
    policy = make_policy(catalog, train_histories)
    policy.save(tmp_path)
    loaded = LearningRecommender.load(tmp_path)
    for h in list(train_histories.values())[:20] + [LearnerHistory()]:
        assert loaded.recommend(h, 10) == policy.recommend(h, 10)


def test_baselines_rank_as_designed(catalog, train_histories):
    n = len(catalog)
    pop = MostPopular().fit(train_histories, n)
    assert top_k(pop.score(LearnerHistory()), set(), 1) == [int(np.argmax(pop.popularity))]
    seq = NextInSequence(catalog).fit(train_histories, n)
    h = history_of(catalog, [0])  # first grammar lesson
    assert top_k(seq.score(h), h.started, 3) == [1, 2, 3]

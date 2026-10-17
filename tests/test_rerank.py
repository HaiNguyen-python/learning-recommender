import numpy as np

from recsys.domain import Catalog, LearnerHistory
from recsys.rerank import RerankConfig, mmr_order, prereqs_met, rerank


def tiny_catalog() -> Catalog:
    """Topic 0: lessons 0 (L1) -> 1 (L2) -> 2 (L3). Topic 1: lessons 3 (L1), 4 (L2), 5 (L1)."""
    return Catalog(
        topics=("grammar", "travel"),
        topic=np.array([0, 0, 0, 1, 1, 1]),
        level=np.array([1, 2, 3, 1, 2, 1]),
        seq=np.array([0, 1, 2, 0, 1, 2]),
        prereqs=((), (0,), (1,), (), (3,), ()),
        ids=("g0", "g1", "g2", "t0", "t1", "t2"),
    )


def done(*lessons: int) -> LearnerHistory:
    h = LearnerHistory()
    for k, i in enumerate(lessons):
        h.add(float(k), i, "view")
        h.add(k + 0.01, i, "complete")
    return h


def test_unmet_prerequisites_are_never_recommended_when_alternatives_exist():
    cat = tiny_catalog()
    base = np.array([0.0, 9.0, 10.0, 0.0, 8.0, 1.0])  # the model loves g2, g1, t1
    h = done(3)  # finished t0 only
    recs = rerank(base, h, cat, est_level=1.5, k=2, cfg=RerankConfig(level_penalty=0.0))
    assert all(prereqs_met(i, h, cat, 1.5) for i in recs)
    assert recs == [4, 5]  # t1 (prereq t0 done), then t2; g1 and g2 are blocked


def test_blocked_lessons_only_fill_a_short_list_and_go_last():
    cat = tiny_catalog()
    h = done(3)
    recs = rerank(np.arange(6.0), h, cat, est_level=1.5, k=5, cfg=RerankConfig(level_penalty=0))
    assert recs[:3] == [5, 4, 0] and set(recs[3:]) == {1, 2}


def test_placement_counts_as_meeting_a_prerequisite():
    cat = tiny_catalog()
    assert not prereqs_met(2, LearnerHistory(), cat, est_level=2.5)
    assert prereqs_met(2, LearnerHistory(), cat, est_level=3.0)  # a level above lesson 1 (L2)


def test_lessons_far_above_level_drop_and_started_lessons_disappear():
    cat = tiny_catalog()
    base = np.array([0.1, 0.0, 1.0, 0.0, 0.0, 0.0])
    h = LearnerHistory()
    cfg = RerankConfig(level_penalty=2.0)
    recs = rerank(base, h, cat, est_level=3.5, k=6, cfg=cfg)
    assert recs[0] == 2  # level 3 is fine for a level-3.5 learner
    recs_beginner = rerank(base, h, cat, est_level=1.0, k=6, cfg=cfg)
    assert recs_beginner.index(2) == len(recs_beginner) - 1
    h.add(0.0, 2, "view")
    assert 2 not in rerank(base, h, cat, est_level=3.5, k=6, cfg=cfg)


def test_mmr_trades_relevance_for_topic_variety():
    scores = np.array([1.0, 0.95, 0.9, 0.5])
    topics = np.array([0, 0, 0, 1])
    assert mmr_order(scores, topics, k=2, lam=1.0) == [0, 1]
    assert mmr_order(scores, topics, k=2, lam=0.6) == [0, 3]

import numpy as np
from scipy import sparse

from recsys.als import ImplicitALS, interaction_matrix
from recsys.domain import LearnerHistory


def block_matrix(seed: int = 0) -> sparse.csr_matrix:
    """Two planted groups: learners 0-9 use lessons 0-9, learners 10-19 use lessons 10-19."""
    rng = np.random.default_rng(seed)
    r = np.zeros((20, 20))
    for u in range(20):
        block = range(0, 10) if u < 10 else range(10, 20)
        for i in block:
            if rng.random() < 0.6:
                r[u, i] = rng.choice([1.0, 3.0])
    return sparse.csr_matrix(r)


def test_loss_never_increases():
    als = ImplicitALS(factors=4, reg=0.1, alpha=5.0, iterations=8).fit_matrix(block_matrix())
    losses = np.array(als.loss_history)
    assert np.all(np.diff(losses) <= 1e-8 * losses[:-1])
    assert losses[-1] < 0.5 * losses[0]


def test_recovers_planted_blocks_for_unseen_lessons():
    r = block_matrix()
    als = ImplicitALS(factors=2, reg=1.0, alpha=5.0, iterations=10).fit_matrix(r)
    scores = als.user_factors @ als.item_factors.T
    for u in range(20):
        seen = set(r[u].indices)
        own = range(0, 10) if u < 10 else range(10, 20)
        other = range(10, 20) if u < 10 else range(0, 10)
        unseen_own = [scores[u, i] for i in own if i not in seen]
        if unseen_own:
            assert min(unseen_own) > max(scores[u, i] for i in other)


def test_fold_in_equals_trained_user_factor():
    """Training ends with a user step, so serving-time fold-in gives the same vector."""
    hists = []
    for u in range(20):
        h = LearnerHistory()
        for k, i in enumerate(range(0, 10) if u < 10 else range(10, 20)):
            if (u + k) % 3:
                h.add(float(k), i, "view")
                if k % 2:
                    h.add(k + 0.01, i, "complete")
        hists.append(h)
    r = interaction_matrix(hists, 20)
    als = ImplicitALS(factors=4, reg=0.5, alpha=3.0, iterations=5).fit_matrix(r)
    for u in (0, 7, 15):
        np.testing.assert_allclose(als.user_vector(hists[u]), als.user_factors[u], atol=1e-10)
    assert np.all(als.user_vector(LearnerHistory()) == 0)


def test_interaction_weights_follow_event_strength():
    h = LearnerHistory()
    h.add(0.0, 2, "view")
    h.add(0.1, 2, "complete")
    h.add(1.0, 5, "view")
    h.add(1.1, 5, "abandon")
    r = interaction_matrix([h], 6).toarray()[0]
    assert r[2] == 3.0 and r[5] == 1.0 and r.sum() == 4.0

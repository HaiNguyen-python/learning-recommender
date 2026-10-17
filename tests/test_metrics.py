import numpy as np
import pytest

from recsys.metrics import (
    average_precision_at_k,
    catalogue_coverage,
    ndcg_at_k,
    novelty,
    recall_at_k,
)


def test_recall_hand_example():
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4) == pytest.approx(2 / 3)
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_hand_example():
    # hits at positions 1 and 3: DCG = 1 + 1/log2(4) = 1.5; ideal (2 hits) = 1 + 1/log2(3)
    assert ndcg_at_k([7, 1, 8], {7, 8}, k=3) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    assert ndcg_at_k([7, 8, 1], {7, 8}, k=3) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {9}, k=2) == 0.0


def test_average_precision_hand_example():
    # hits at 1 and 3: (1/1 + 2/3) / min(3, k=3)
    assert average_precision_at_k([5, 0, 6], {5, 6, 9}, k=3) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision_at_k([5, 6], {5, 6}, k=10) == pytest.approx(1.0)


def test_coverage_and_novelty():
    assert catalogue_coverage([[0, 1], [1, 2]], n_items=10) == pytest.approx(0.3)
    bits = np.array([0.0, 1.0, 3.0])
    assert novelty([[0, 2], [2]], bits) == pytest.approx(2.0)
    assert novelty([], bits) == 0.0

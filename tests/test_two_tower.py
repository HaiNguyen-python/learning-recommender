import pytest

torch = pytest.importorskip("torch")  # only with the [deep] extra


def test_two_tower_scores_every_lesson(catalog, train_histories):
    from recsys.two_tower import TwoTower

    model = TwoTower(catalog, dim=8, epochs=1).fit(train_histories, len(catalog))
    h = next(iter(train_histories.values()))
    assert model.score(h).shape == (len(catalog),)

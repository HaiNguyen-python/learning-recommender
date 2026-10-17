import pytest

from recsys.domain import histories_from_events
from recsys.simulator import generate_history, make_world

N_LEARNERS, N_DAYS, CUTOFF = 250, 28, 21.0


@pytest.fixture(scope="session")
def world():
    """250 learners x 28 days: small enough for CI, big enough for the planted effects."""
    return make_world(N_LEARNERS, N_DAYS, seed=3)


@pytest.fixture(scope="session")
def history(world):
    return generate_history(world, N_DAYS, seed=4)


@pytest.fixture(scope="session")
def events(history):
    return history[0]


@pytest.fixture(scope="session")
def catalog(world):
    return world.catalog


@pytest.fixture(scope="session")
def train_histories(events, catalog):
    return histories_from_events(events, catalog, CUTOFF)

import pytest
from fastapi.testclient import TestClient

from recsys.api import create_app
from recsys.artifacts import load_artifacts, train_artifacts
from recsys.evaluate import ALSConfig, OfflineConfig


@pytest.fixture(scope="module")
def artefacts(tmp_path_factory, events, catalog):
    out = tmp_path_factory.mktemp("artifacts")
    meta = train_artifacts(events, catalog, 28.0, out, OfflineConfig(als=ALSConfig(iterations=4)))
    assert meta["feature_parity_max_diff"] == 0.0
    return out


@pytest.fixture(scope="module")
def client(artefacts):
    return TestClient(create_app(load_artifacts(artefacts)))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["lessons"] == 300 and body["learners"] > 100


def test_known_learner_gets_k_unseen_lessons_with_met_prerequisites(client, events):
    learner = events.groupby("learner_id").size().idxmax()
    body = client.get("/recommend", params={"learner_id": learner, "k": 7}).json()
    assert body["known_learner"] and body["strategy"] == "ALS (implicit)"
    assert len(body["items"]) == 7
    seen = set(events.loc[events["learner_id"] == learner, "lesson_id"])
    assert not {x["lesson_id"] for x in body["items"]} & seen
    assert all(x["prereqs_met"] for x in body["items"])
    assert body["features"]["n_started"] == len(seen)


def test_unknown_learner_gets_the_cold_start_fallback(client):
    body = client.get("/recommend", params={"learner_id": "new-learner"}).json()
    assert not body["known_learner"] and body["strategy"].startswith("fallback")
    assert len(body["items"]) == 10 and max(x["level"] for x in body["items"]) <= 2


@pytest.mark.parametrize("params", [{"learner_id": "L0001", "k": 0}, {"k": 5}])
def test_invalid_requests_are_rejected(client, params):
    assert client.get("/recommend", params=params).status_code == 422

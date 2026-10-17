import pytest

from recsys.data import catalog_from_events, load_ednet_kt1, load_implicit_csv
from recsys.domain import histories_from_events
from recsys.evaluate import ALSConfig, OfflineConfig, run_offline

CSV = """user_id,item_id,timestamp
u1,v10,2024-01-01T10:00:00Z
u1,v11,2024-01-02T10:00:00Z
u2,v10,2024-01-01T12:00:00Z
u2,v12,2024-01-03T09:00:00Z
u3,v11,2024-01-04T09:00:00Z
u3,v10,2024-01-05T09:00:00Z
"""

KT1 = """timestamp,solving_id,question_id,user_answer,elapsed_time
1565096190868,1,q5012,b,38000
1565096221062,2,q4706,c,24000
1565096293432,3,q4366,a,35000
"""


def test_generic_csv_maps_to_canonical_events(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text(CSV)
    ev = load_implicit_csv(path)
    assert list(ev.columns) == ["learner_id", "lesson_id", "t", "event"]
    assert ev["t"].min() == 0.0 and ev["t"].max() == pytest.approx(3 + 23 / 24)
    assert set(ev["event"]) == {"view"}
    cat = catalog_from_events(ev)
    hist = histories_from_events(ev, cat)
    assert hist["u2"].started == {cat.index()["v10"], cat.index()["v12"]}


def test_unknown_event_types_are_rejected(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("user_id,item_id,timestamp,kind\na,b,1,clicked\n")
    with pytest.raises(ValueError, match="clicked"):
        load_implicit_csv(path, event_col="kind")


def test_ednet_kt1_folder(tmp_path):
    (tmp_path / "u1.csv").write_text(KT1)
    (tmp_path / "u2.csv").write_text(KT1.replace("q4366", "q9999"))
    ev = load_ednet_kt1(tmp_path)
    assert len(ev) == 12 and set(ev["learner_id"]) == {"u1", "u2"}
    assert (ev.groupby("event").size() == 6).all()
    with pytest.raises(FileNotFoundError):
        load_ednet_kt1(tmp_path / "missing")


def test_offline_evaluation_runs_on_a_loaded_log(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text(CSV)
    ev = load_implicit_csv(path)
    ev.loc[ev["t"] > 2.5, "event"] = "complete"
    result = run_offline(
        ev, catalog_from_events(ev), 2.5, OfflineConfig(als=ALSConfig(factors=2, iterations=2))
    )
    assert result.table.loc[("Most popular", "all"), "n_learners"] == 1.0

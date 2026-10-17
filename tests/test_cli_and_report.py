import json

from recsys.abtest import ABConfig
from recsys.cli import main
from recsys.evaluate import ALSConfig, OfflineConfig
from recsys.report import ReportConfig, write

SMALL = ReportConfig(
    n_learners=200,
    n_days=21,
    cutoff_day=15,
    offline=OfflineConfig(als=ALSConfig(iterations=3)),
    ab=ABConfig(n_days=7, pre_days=7),
)


def test_report_writes_results_and_charts(tmp_path):
    study = write(tmp_path, SMALL)
    text = (tmp_path / "results.md").read_text()
    assert "Synthetic data" in text and "CUPED" in text and "cold start" in text.lower()
    assert f"{study.ab.primary.diff:+.3f}" in text
    for name in ("offline_recall.png", "ab_effect.png"):
        assert (tmp_path / "img" / name).stat().st_size > 10_000
    assert study.parity == 0.0


def test_cli_generate_train_and_evaluate(tmp_path, capsys):
    args = ["--learners", "60", "--days", "8"]
    main(["generate", "--out", str(tmp_path / "data"), *args])
    assert (tmp_path / "data" / "events.csv").exists()
    main(["train", "--artifacts", str(tmp_path / "art"), *args])
    meta = json.loads((tmp_path / "art" / "meta.json").read_text())
    assert meta["n_lessons"] == 300 and meta["feature_parity_max_diff"] == 0.0
    main(["evaluate", "--data", str(tmp_path / "data" / "events.csv")])
    assert "Most popular" in capsys.readouterr().out

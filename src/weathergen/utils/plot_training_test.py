from pathlib import Path

from weathergen.utils.plot_training import filter_runs_with_metrics


def test_filter_runs_with_metrics_splits_available_and_missing(tmp_path: Path):
    results_root = tmp_path / "results"
    available_metrics = results_root / "run-a" / "run-a_train_metrics.json"
    available_metrics.parent.mkdir(parents=True)
    available_metrics.write_text("")

    runs_ids = {
        "run-a": [0, "available"],
        "run-b": [0, "missing"],
    }

    available, missing = filter_runs_with_metrics(runs_ids, results_root=results_root)

    assert available == {"run-a": [0, "available"]}
    assert missing == {"run-b": results_root / "run-b" / "run-b_train_metrics.json"}

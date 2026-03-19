from types import SimpleNamespace

import polars as pl
import pytest

from weathergen.utils import train_logger


class DummyConfig(dict):
    def __getattr__(self, key):
        return self[key]


def test_train_logger_read_does_not_require_legacy_text_logs(monkeypatch, tmp_path):
    run_id = "abc123"
    run_dir = tmp_path / run_id

    cf = DummyConfig(
        general=SimpleNamespace(run_id=run_id),
        training_config={},
        validation_config={},
        streams=[],
    )

    def fail_open(*_args, **_kwargs):
        raise AssertionError("TrainLogger.read should not open legacy text log files.")

    monkeypatch.setattr(train_logger.config, "load_merge_configs", lambda **_kwargs: cf)
    monkeypatch.setattr(train_logger.config, "get_path_run", lambda _cf: run_dir)
    monkeypatch.setattr(
        train_logger,
        "get_active_stage_config",
        lambda *_args, **_kwargs: {"losses": {}},
    )
    monkeypatch.setattr(train_logger, "get_loss_terms_per_stream", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(train_logger, "read_metrics", lambda *_args, **_kwargs: pl.DataFrame())
    monkeypatch.setattr("builtins.open", fail_open)

    metrics = train_logger.TrainLogger.read(run_id)

    assert metrics.run_id == run_id


def test_read_metrics_raises_plain_file_not_found(tmp_path):
    cf = DummyConfig(general=SimpleNamespace(run_id="abc123"))

    with pytest.raises(FileNotFoundError, match="abc123_train_metrics.json"):
        train_logger.read_metrics(
            cf=cf,
            run_id="abc123",
            stage="train",
            cols=["loss_avg_mean"],
            cols_patterns=[],
            results_path=tmp_path / "results",
        )

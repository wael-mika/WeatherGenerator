from io import StringIO
from math import isnan
from pathlib import Path

from weathergen.utils.metrics import (
    get_train_metrics_path,
    read_metrics_file,
)

s = """{"weathergen.timestamp":100, "m": "nan"}
{"weathergen.timestamp":101,"m": 1.3}
{"weathergen.timestamp":102,"a": 4}
"""


def test1():
    df = read_metrics_file(StringIO(s))
    assert df.shape == (3, 3)
    assert df["weathergen.timestamp"].to_list() == [100, 101, 102]
    assert isnan(df["m"].to_list()[0])
    assert df["m"].to_list()[1:] == [1.3, None]
    assert df["a"].to_list() == [None, None, 4]


def test_get_train_metrics_path_prefers_run_dir_new_layout(tmp_path: Path):
    run_id = "abc123"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    expected = run_dir / f"{run_id}_train_metrics.json"
    expected.write_text("")

    assert get_train_metrics_path(tmp_path / "results", run_id) == expected


def test_get_train_metrics_path_supports_run_dir_legacy_layout(tmp_path: Path):
    run_id = "abc123"
    run_dir = tmp_path / "results" / run_id
    run_dir.mkdir(parents=True)
    expected = run_dir / "metrics.json"
    expected.write_text("")

    assert get_train_metrics_path(run_dir, run_id) == expected


def test_get_train_metrics_path_defaults_to_current_layout(tmp_path: Path):
    run_id = "abc123"
    expected = tmp_path / "results" / run_id / f"{run_id}_train_metrics.json"

    assert get_train_metrics_path(tmp_path / "results", run_id) == expected

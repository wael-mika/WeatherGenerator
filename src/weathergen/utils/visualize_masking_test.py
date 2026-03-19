from types import SimpleNamespace

from weathergen.train.utils import TRAIN, VAL
from weathergen.utils.visualize_masking import _get_stream_plot_mode, _resolve_stream_views


def _make_stream(name: str, **values):
    return SimpleNamespace(name=name, get=lambda key, default=None: values.get(key, default))


def test_get_stream_plot_mode_forcing_stream_is_source_only():
    stream = _make_stream(
        "ERA5",
        forcing=True,
        train_source_channels=["2t"],
        train_target_channels=[],
    )

    assert _get_stream_plot_mode(stream, TRAIN) == "source_only"


def test_get_stream_plot_mode_diagnostic_stream_is_target_only():
    stream = _make_stream(
        "METOPBIASI",
        diagnostic=True,
        val_source_channels=[],
        val_target_channels=["obsvalue_rawbt_16"],
    )

    assert _get_stream_plot_mode(stream, VAL) == "target_only"


def test_get_stream_plot_mode_regular_stream_is_both():
    stream = _make_stream(
        "SURFACE",
        train_source_channels=["t2m"],
        train_target_channels=["t2m"],
    )

    assert _get_stream_plot_mode(stream, TRAIN) == "both"


def test_resolve_stream_views_pairs_auto_selected_diagnostic_with_forcing_source():
    streams = [
        _make_stream("ERA5", forcing=True, train_source_channels=["2t"], train_target_channels=[]),
        _make_stream(
            "METOPBIASI",
            diagnostic=True,
            train_source_channels=[],
            train_target_channels=["obsvalue_rawbt_16"],
        ),
    ]

    resolved = _resolve_stream_views(streams, None, TRAIN)

    assert resolved["selected_stream"].name == "METOPBIASI"
    assert resolved["source_stream"].name == "ERA5"
    assert resolved["target_stream"].name == "METOPBIASI"
    assert resolved["paired_streams"] is True


def test_resolve_stream_views_pairs_explicit_forcing_with_diagnostic_target():
    streams = [
        _make_stream("ERA5", forcing=True, train_source_channels=["2t"], train_target_channels=[]),
        _make_stream(
            "METOPBIASI",
            diagnostic=True,
            train_source_channels=[],
            train_target_channels=["obsvalue_rawbt_16"],
        ),
    ]

    resolved = _resolve_stream_views(streams, "ERA5", TRAIN)

    assert resolved["selected_stream"].name == "ERA5"
    assert resolved["source_stream"].name == "ERA5"
    assert resolved["target_stream"].name == "METOPBIASI"
    assert resolved["paired_streams"] is True


def test_resolve_stream_views_keeps_regular_stream_as_single_stream_view():
    streams = [
        _make_stream("SURFACE", train_source_channels=["t2m"], train_target_channels=["t2m"]),
        _make_stream("ERA5", forcing=True, train_source_channels=["2t"], train_target_channels=[]),
    ]

    resolved = _resolve_stream_views(streams, "SURFACE", TRAIN)

    assert resolved["source_stream"].name == "SURFACE"
    assert resolved["target_stream"].name == "SURFACE"
    assert resolved["paired_streams"] is False

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Lightweight dict/config utility functions."""

from collections import defaultdict

import omegaconf as oc


def nested_dict():
    """Two-level nested dict factory: dict[key1][key2] = value"""
    return defaultdict(dict)


def triple_nested_dict():
    """Three-level nested dict factory: dict[key1][key2][key3] = value"""
    return defaultdict(nested_dict)


def merge(dst: dict, src: dict) -> dict:
    """Recursively merge *src* into *dst*. Values in *src* overwrite *dst*.

    Parameters
    ----------
    dst : dict
        Destination dictionary.
    src : dict
        Source dictionary.

    Returns
    -------
    dict
        Merged dictionary (same object as *dst*).
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = v
    return dst


#: Separator between a metric name and its threshold in an expanded metric name.
THRESHOLD_SUFFIX = "_thr"


def format_threshold(value) -> str:
    """Render a threshold as the compact token used in expanded metric names.

    ``%g`` keeps the token short and stable (``0.001`` rather than ``0.001000``), which
    matters because the token ends up in the score JSON filename
    (``<run>_<stream>_<region>_<metric>_chkpt00000.json``) and therefore acts as a cache key.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:g}"


def base_metric_name(metric: str) -> str:
    """Strip the ``_thr<value>`` suffix that :func:`parse_metric_params` appends.

    Expanded names such as ``ets_thr0.001`` must still resolve to the ``ets`` scoring
    function, to the correct argument list, and to the right "is lower better?" answer,
    so every lookup keyed by metric name goes through this helper. Names without the
    suffix are returned unchanged, and no metric in the registries contains ``_thr``.
    """
    head, sep, _ = metric.partition(THRESHOLD_SUFFIX)
    return head if sep else metric


def _expand_threshold_lists(metrics: oc.DictConfig) -> oc.DictConfig:
    """Expand list-valued ``thresh`` entries into one metric per threshold.

    ``{'ets': {'thresh': [0.001, 0.02]}}`` becomes
    ``{'ets_thr0.001': {'thresh': 0.001}, 'ets_thr0.02': {'thresh': 0.02}}``.

    The expansion is needed because a metric name is the key of the metrics mapping, the
    label of the ``metric`` coordinate, and part of the score JSON filename -- so each
    threshold needs a name of its own to be scored, cached and plotted independently.
    A scalar ``thresh`` is left untouched, which keeps existing configs and their cached
    score files valid.
    """
    expanded = oc.DictConfig({})
    for name, params in metrics.items():
        params_d = oc.OmegaConf.to_container(params, resolve=True) if params is not None else {}
        if not isinstance(params_d, dict):
            params_d = {}
        thresh = params_d.get("thresh")

        if not isinstance(thresh, list | tuple):
            expanded = oc.OmegaConf.merge(expanded, {name: params_d})
            continue

        if len(thresh) == 0:
            raise ValueError(
                f"Metric '{name}' has an empty 'thresh' list; give at least one value."
            )

        seen: dict[str, object] = {}
        for value in thresh:
            token = format_threshold(value)
            if token in seen:
                raise ValueError(
                    f"Metric '{name}' has thresholds {seen[token]!r} and {value!r} that both "
                    f"render as '{token}'. They would collide in the score filename; "
                    f"remove the duplicate."
                )
            seen[token] = value
            expanded = oc.OmegaConf.merge(
                expanded, {f"{name}{THRESHOLD_SUFFIX}{token}": {**params_d, "thresh": value}}
            )
    return expanded


def parse_metric_params(metrics) -> oc.DictConfig:
    """Convert a mixed list of str/dict metrics into a ``{name: params}`` DictConfig.

    The config may look like::

        metrics:
          - fbi:
              thresh: 280
          - rmse

    In Python that becomes ``[{'fbi': {'thresh': 280}}, 'rmse']``.
    This function converts it to ``{'fbi': {'thresh': 280}, 'rmse': {}}``.

    A ``thresh`` given as a list is expanded into one metric per value -- see
    :func:`_expand_threshold_lists` -- so::

        metrics:
          - ets:
              thresh: [0.001, 0.02]

    yields ``{'ets_thr0.001': {'thresh': 0.001}, 'ets_thr0.02': {'thresh': 0.02}}``.
    """
    out = oc.DictConfig({})
    for metric in metrics:
        if isinstance(metric, str):
            out = oc.OmegaConf.merge(out, {metric: {}})
        else:
            out = oc.OmegaConf.merge(out, metric)
    return _expand_threshold_lists(out)

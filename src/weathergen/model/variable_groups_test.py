# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

import pytest

from weathergen.model.utils import _resolve_variable_groups


class TestResolveVariableGroups:
    def _simple_cfg(self):
        return {
            "dynamics": {
                "variables": [r"u_\d+", r"v_\d+", r"z_\d+"],
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
            "_default": {
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
        }

    def test_indices_correctly_assigned(self):
        channels = ["u_850", "v_850", "z_500", "tp", "2t"]
        groups = _resolve_variable_groups(self._simple_cfg(), channels)
        by_name = {g[0]: g[1] for g in groups}
        assert by_name["dynamics"] == [0, 1, 2]
        assert set(by_name["_default"]) == {3, 4}

    def test_default_collects_remainder(self):
        channels = ["u_500", "tp", "sp"]
        groups = _resolve_variable_groups(self._simple_cfg(), channels)
        by_name = {g[0]: g[1] for g in groups}
        assert by_name["dynamics"] == [0]
        assert set(by_name["_default"]) == {1, 2}

    def test_overlap_raises(self):
        cfg = {
            "a": {"variables": [r"u_\d+"], "pred_head": {}},
            "b": {"variables": ["u_850"], "pred_head": {}},
            "_default": {"pred_head": {}},
        }
        with pytest.raises(ValueError, match="multiple variable groups"):
            _resolve_variable_groups(cfg, ["u_850", "tp"])

    def test_missing_default_raises_when_unmatched(self):
        cfg = {"dynamics": {"variables": [r"u_\d+"], "pred_head": {}}}
        with pytest.raises(ValueError, match="_default"):
            _resolve_variable_groups(cfg, ["u_850", "tp"])

    def test_no_default_needed_when_all_matched(self):
        cfg = {"all": {"variables": [r"u_\d+", "tp"], "pred_head": {}}}
        groups = _resolve_variable_groups(cfg, ["u_850", "tp"])
        assert len(groups) == 1
        assert groups[0][0] == "all"
        assert groups[0][1] == [0, 1]

    def test_empty_channels_returns_only_default(self):
        cfg = {
            "dynamics": {"variables": [r"u_\d+"], "pred_head": {}},
            "_default": {"pred_head": {}},
        }
        groups = _resolve_variable_groups(cfg, [])
        by_name = {g[0]: g[1] for g in groups}
        assert by_name["dynamics"] == []
        assert by_name["_default"] == []

    def test_fullmatch_not_partial(self):
        cfg = {
            "upper": {"variables": [r"u_\d+"], "pred_head": {}},
            "_default": {"pred_head": {}},
        }
        # "u_extra_850" should NOT match "u_\d+" (fullmatch required)
        channels = ["u_850", "u_extra_850"]
        groups = _resolve_variable_groups(cfg, channels)
        by_name = {g[0]: g[1] for g in groups}
        assert by_name["upper"] == [0]
        assert by_name["_default"] == [1]


class TestOptionBConfig:
    """Tests for Option B (per-group target_readout) config plumbing through the resolver."""

    def test_target_readout_preserved_in_group_cfg(self):
        cfg = {
            "dynamics": {
                "variables": [r"u_\d+", r"v_\d+"],
                "target_readout": {"num_layers": 2, "num_heads": 8},
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
            "_default": {
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Softplus"},
            },
        }
        groups = _resolve_variable_groups(cfg, ["u_850", "v_850", "tp"])
        by_cfg = {g[0]: g[2] for g in groups}

        # dynamics group carries target_readout → model.py will set has_own_tte=True
        assert "target_readout" in by_cfg["dynamics"]
        assert by_cfg["dynamics"]["target_readout"]["num_heads"] == 8
        assert by_cfg["dynamics"]["target_readout"]["num_layers"] == 2

        # _default has no target_readout → model.py will set has_own_tte=False
        assert "target_readout" not in by_cfg["_default"]

    def test_mixed_option_a_and_b(self):
        cfg = {
            "precip": {
                "variables": ["tp", "cp"],
                "target_readout": {"num_layers": 1, "num_heads": 4},
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Softplus"},
            },
            "upper": {
                "variables": [r"u_\d+", r"v_\d+"],
                # no target_readout → Option A (shares stream TTE)
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
            "_default": {
                "pred_head": {"ens_size": 1, "num_layers": 1, "final_activation": "Identity"},
            },
        }
        channels = ["u_850", "v_850", "tp", "cp", "2t"]
        groups = _resolve_variable_groups(cfg, channels)
        by_name = {g[0]: g for g in groups}

        # Indices
        assert by_name["precip"][1] == [2, 3]
        assert by_name["upper"][1] == [0, 1]
        assert by_name["_default"][1] == [4]

        # has_own_tte flag (detected by model.py as "target_readout" in group_cfg)
        has_own_tte = {name: "target_readout" in cfg for name, _, cfg in groups}
        assert has_own_tte["precip"] is True
        assert has_own_tte["upper"] is False
        assert has_own_tte["_default"] is False

    def test_needs_shared_tte_when_any_group_lacks_target_readout(self):
        cfg = {
            "a": {
                "variables": [r"u_\d+"],
                "target_readout": {"num_layers": 1, "num_heads": 4},
                "pred_head": {},
            },
            "_default": {"pred_head": {}},  # no target_readout → needs shared TTE
        }
        groups = _resolve_variable_groups(cfg, ["u_850", "tp"])
        needs_shared = any("target_readout" not in gcfg for _, _, gcfg in groups)
        assert needs_shared is True

    def test_no_shared_tte_when_all_groups_have_target_readout(self):
        cfg = {
            "dynamics": {
                "variables": [r"u_\d+"],
                "target_readout": {"num_layers": 2, "num_heads": 4},
                "pred_head": {},
            },
            "_default": {
                "target_readout": {"num_layers": 1, "num_heads": 4},
                "pred_head": {},
            },
        }
        groups = _resolve_variable_groups(cfg, ["u_850", "tp"])
        needs_shared = any("target_readout" not in gcfg for _, _, gcfg in groups)
        assert needs_shared is False

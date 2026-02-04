"""Tests for masking strategy/relationship validation."""

import pytest

from weathergen.datasets.masking_utils import (
    DEFAULT_RELATIONSHIP,
    INVALID_COMBINATIONS,
    SOURCE_IGNORED_RELATIONSHIPS,
    VALID_RELATIONSHIPS,
    VALID_STRATEGIES,
    check_masking_config,
    parse_source_target_mapping,
    validate_masking_config,
)


class TestMaskingConstants:
    """Test masking constants are properly defined."""

    def test_valid_strategies_defined(self):
        expected = {"random", "healpix", "cropping_healpix", "forecast", "causal"}
        assert VALID_STRATEGIES == expected

    def test_valid_relationships_defined(self):
        expected = {"independent", "complement", "identity", "subset", "disjoint"}
        assert VALID_RELATIONSHIPS == expected

    def test_source_config_ignored_relationships(self):
        assert "complement" in SOURCE_IGNORED_RELATIONSHIPS
        assert "identity" in SOURCE_IGNORED_RELATIONSHIPS
        assert "independent" not in SOURCE_IGNORED_RELATIONSHIPS

    def test_invalid_forecast_combinations(self):
        for rel in VALID_RELATIONSHIPS:
            if rel != "independent":
                assert ("forecast", rel) in INVALID_COMBINATIONS

    def test_default_relationships(self):
        assert DEFAULT_RELATIONSHIP["random"] == "complement"
        assert DEFAULT_RELATIONSHIP["healpix"] == "independent"
        assert DEFAULT_RELATIONSHIP["cropping_healpix"] == "independent"


class TestParseSourceTargetMapping:
    """Test parse_source_target_mapping helper function."""

    def test_default_mapping_when_no_losses(self):
        mapping = parse_source_target_mapping({}, num_sources=3, num_targets=3)
        assert mapping[0] == (None, 0)
        assert mapping[1] == (None, 1)
        assert mapping[2] == (None, 2)

    def test_default_mapping_more_sources_than_targets(self):
        mapping = parse_source_target_mapping({}, num_sources=3, num_targets=1)
        assert mapping[0] == (None, 0)
        assert mapping[1] == (None, 0)
        assert mapping[2] == (None, 0)

    def test_explicit_relationship_from_losses(self):
        losses = {
            "test_loss": {
                "loss_fcts": {
                    "loss1": {
                        "target_source_correspondence": {0: {0: "subset", 1: "complement"}}
                    }
                }
            }
        }
        mapping = parse_source_target_mapping(losses, num_sources=2, num_targets=1)
        assert mapping[0] == ("subset", 0)
        assert mapping[1] == ("complement", 0)


class TestValidateMaskingConfig:
    """Test validate_masking_config function."""

    def test_forecast_with_complement_raises_error(self):
        source_cfgs = [{"masking_strategy": "forecast"}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "complement"}}}}}}

        with pytest.raises(ValueError, match="forecast.*incompatible.*complement"):
            validate_masking_config(source_cfgs, target_cfgs, losses)

    def test_forecast_with_independent_valid(self):
        source_cfgs = [{"masking_strategy": "forecast"}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "independent"}}}}}}

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses)
        assert len(warnings) == 0

    def test_random_with_complement_warns(self):
        source_cfgs = [{"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "complement"}}}}}}

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses, strict=False)
        assert len(warnings) == 1
        assert "IGNORED" in warnings[0]

    def test_random_with_complement_strict_raises(self):
        source_cfgs = [{"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "complement"}}}}}}

        with pytest.raises(ValueError, match="Strict"):
            validate_masking_config(source_cfgs, target_cfgs, losses, strict=True)

    def test_random_with_subset_valid(self):
        source_cfgs = [{"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "subset"}}}}}}

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses)
        assert len(warnings) == 0

    def test_default_relationship_for_random_warns(self):
        source_cfgs = [{"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: 0}}}}}

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses)
        assert len(warnings) == 1
        assert "default" in warnings[0]

    def test_cropping_healpix_with_complement_warns_not_contiguous(self):
        source_cfgs = [{"masking_strategy": "cropping_healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        target_cfgs = [{"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}]
        losses = {"test": {"loss_fcts": {"test_loss": {"target_source_correspondence": {0: {0: "complement"}}}}}}

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses)
        assert len(warnings) == 1
        assert "NOT spatially contiguous" in warnings[0]

    def test_empty_source_cfgs_returns_empty_warnings(self):
        warnings = validate_masking_config([], [], {})
        assert warnings == []

    def test_multiple_sources_validates_each(self):
        source_cfgs = [
            {"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}},
            {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}},
        ]
        target_cfgs = [
            {"masking_strategy": "random", "masking_strategy_config": {"rate": 0.5}},
            {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.3, "hl_mask": 0}},
        ]
        losses = {
            "test": {
                "loss_fcts": {
                    "test_loss": {"target_source_correspondence": {0: {0: "complement"}, 1: {1: "complement"}}}
                }
            }
        }

        warnings = validate_masking_config(source_cfgs, target_cfgs, losses)
        assert len(warnings) == 2
        assert "source[0]" in warnings[0]
        assert "source[1]" in warnings[1]


class TestCheckMaskingConfig:
    """Test check_masking_config function for pre-training validation."""

    def test_valid_config_returns_true(self):
        config = {
            "model_input": [
                {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
            ],
            "target_input": [
                {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
            ],
            "losses": {
                "test": {"loss_fcts": {"loss1": {"target_source_correspondence": {0: {0: "independent"}}}}}
            },
        }
        is_valid, warnings, errors = check_masking_config(config, print_summary=False)
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_invalid_config_returns_false(self):
        config = {
            "model_input": [{"masking_strategy": "forecast"}],
            "target_input": [
                {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
            ],
            "losses": {
                "test": {"loss_fcts": {"loss1": {"target_source_correspondence": {0: {0: "complement"}}}}}
            },
        }
        is_valid, warnings, errors = check_masking_config(config, print_summary=False)
        assert is_valid is False
        assert len(errors) == 1
        assert "forecast" in errors[0]

    def test_config_with_warnings(self):
        config = {
            "model_input": [
                {"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}
            ],
            "target_input": [
                {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
            ],
            "losses": {
                "test": {"loss_fcts": {"loss1": {"target_source_correspondence": {0: {0: "complement"}}}}}
            },
        }
        is_valid, warnings, errors = check_masking_config(config, print_summary=False)
        assert is_valid is True
        assert len(warnings) == 1
        assert "IGNORED" in warnings[0]

    def test_strict_mode_fails_on_warnings(self):
        config = {
            "model_input": [
                {"masking_strategy": "random", "masking_strategy_config": {"rate": 0.6}}
            ],
            "target_input": [
                {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
            ],
            "losses": {
                "test": {"loss_fcts": {"loss1": {"target_source_correspondence": {0: {0: "complement"}}}}}
            },
        }
        is_valid, warnings, errors = check_masking_config(config, strict=True, print_summary=False)
        assert is_valid is False
        assert len(errors) == 1

    def test_nested_stage_config(self):
        config = {
            "stage": {
                "model_input": [
                    {"masking_strategy": "healpix", "masking_strategy_config": {"rate": 0.4, "hl_mask": 0}}
                ],
                "losses": {},
            }
        }
        is_valid, warnings, errors = check_masking_config(config, print_summary=False)
        assert is_valid is True

    def test_empty_config(self):
        config = {}
        is_valid, warnings, errors = check_masking_config(config, print_summary=False)
        assert is_valid is True
        assert len(warnings) == 0
        assert len(errors) == 0

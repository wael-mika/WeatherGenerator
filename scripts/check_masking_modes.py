"""Standalone check for the three masking modes in build_samples_for_stream.

Run from the WeatherGenerator repo root:
    uv run --offline python scripts/check_masking_modes.py
"""
import numpy as np
from omegaconf import OmegaConf

from weathergen.datasets.masking import Masker

# ── helpers ──────────────────────────────────────────────────────────────────
HL = 5                      # healpix level 5 → 12 * 4^5 = 12288 cells
NUM_CELLS = 12 * (4 ** HL)
RNG = np.random.default_rng(42)

LOSSES_CFG = OmegaConf.create({
    "physical": {
        "loss_fcts": {"mse": {}}
    }
})

# is_stream_diagnostic checks train_source_channels (absent/empty → diagnostic).
# is_stream_forcing  checks train_target_channels (absent/empty → forcing).
# Both must be non-empty to make a plain (non-diagnostic, non-forcing) stream.
ERA5_STREAM = {"name": "ERA5", "train_source_channels": ["t2m"], "train_target_channels": ["t2m"]}


def mask_np(mask_data, idx=0):
    m = mask_data.masks[idx]
    return m.numpy() if hasattr(m, "numpy") else np.array(m)


def make_masker(mode_cfg, stream_info):
    masker = Masker(healpix_level=HL, stage="train",
                    streams=[stream_info], mode_cfg=mode_cfg)
    masker.reset_rng(RNG)
    return masker


def run(masker, stream_info):
    return masker.build_samples_for_stream(
        training_mode="masking",
        num_cells=NUM_CELLS,
        stream_info=stream_info,
    )


# ── Mode A: MAE pretraining ───────────────────────────────────────────────────
print("=== Mode A: MAE (random masking, no target_input) ===")
mode_a_cfg = OmegaConf.create({
    "model_input": {
        "mae_random": {
            "masking_strategy": "random",
            "num_samples": 1,
            "masking_strategy_config": {"rate": 0.15, "rate_sampling": False},
        }
    },
    "losses": LOSSES_CFG,
})
masker_a = make_masker(mode_a_cfg, ERA5_STREAM)
tgt_a, src_a, _ = run(masker_a, ERA5_STREAM)

assert len(src_a) == 1, "Mode A: expected exactly 1 source sample"
assert len(tgt_a) == 1, "Mode A: expected exactly 1 target sample"
src_m = mask_np(src_a)
tgt_m = mask_np(tgt_a)
assert not np.any(src_m & tgt_m), "Mode A: source and target must not overlap"
assert np.all(src_m | tgt_m),     "Mode A: source ∪ target must cover all cells"
print("  PASS: source ∩ target = ∅, source ∪ target = all cells")


# ── Mode B: Forecast finetuning (clean config) ────────────────────────────────
print("=== Mode B: Forecast finetuning (mae_random disabled via num_samples=0) ===")
mode_b_clean_cfg = OmegaConf.create({
    "model_input": {
        "mae_random":  {"masking_strategy": "random",   "num_samples": 0},
        "forecasting": {"masking_strategy": "forecast", "num_samples": 1},
    },
    "losses": LOSSES_CFG,
})
masker_b = make_masker(mode_b_clean_cfg, ERA5_STREAM)
tgt_b, src_b, _ = run(masker_b, ERA5_STREAM)

assert len(src_b) == 1, "Mode B (clean): expected exactly 1 source sample"
assert len(tgt_b) == 1, "Mode B (clean): expected exactly 1 target sample"
assert np.all(mask_np(src_b)), "Mode B (clean): source mask must be all-True"
assert np.all(mask_np(tgt_b)), "Mode B (clean): target mask must be all-True"
print("  PASS: 1 source, 1 target, both all-True")


# ── Mode B: Forecast finetuning (simulate config-merge failure) ───────────────
print("=== Mode B: Forecast finetuning (merge failure: mae_swath.num_samples=1) ===")
mode_b_fail_cfg = OmegaConf.create({
    "model_input": {
        # Simulates config-merge failure: num_samples stayed at 1 (should be 0)
        "mae_swath": {
            "masking_strategy": "satellite_swath",
            "num_samples": 1,
            "masking_strategy_config": {
                "num_swaths": 4,
                "swath_width_deg": 15.0,
                "orbit_drift_deg": -25.0,
            },
        },
        "forecasting": {"masking_strategy": "forecast", "num_samples": 1},
    },
    "losses": LOSSES_CFG,
})
masker_bf = make_masker(mode_b_fail_cfg, ERA5_STREAM)
tgt_bf, src_bf, _ = run(masker_bf, ERA5_STREAM)

assert len(src_bf) == 1, "Mode B (merge fail): expected exactly 1 source sample (swath filtered)"
assert len(tgt_bf) == 1, "Mode B (merge fail): expected exactly 1 target sample"
assert np.all(mask_np(src_bf)), "Mode B (merge fail): source must be all-True (forecast only)"
assert np.all(mask_np(tgt_bf)), "Mode B (merge fail): target must be all-True (forecast only)"
print("  PASS: swath suppressed despite num_samples=1; 1 all-True pair produced")


# ── Mode C: IASI cross-stream finetuning ──────────────────────────────────────
print("=== Mode C: Explicit target (ERA5 forcing / IASI diagnostic) ===")
mode_c_cfg = OmegaConf.create({
    "model_input": {
        "mae_random_disabled": {"masking_strategy": "random", "num_samples": 0},
        "input": {
            "masking_strategy": "random",
            "num_samples": 1,
            "masking_strategy_config": {"rate": 1.0, "rate_sampling": False},
        },
    },
    "target_input": {
        "mae_target_disabled": {"masking_strategy": "random", "num_samples": 0},
        "target": {
            "masking_strategy": "random",
            "num_samples": 1,
            "masking_strategy_config": {"rate": 1.0, "rate_sampling": False},
        },
    },
    "losses": LOSSES_CFG,
})

# ERA5: forcing=True → target=all-False, source=complement(all-False)=all-True
# is_stream_forcing checks the "forcing" key; train_source_channels is still needed
# so is_stream_diagnostic does not also fire (would zero the source too).
ERA5_FORCING = {"name": "ERA5", "forcing": True, "train_source_channels": ["t2m"], "train_target_channels": ["t2m"]}
masker_c_era5 = make_masker(mode_c_cfg, ERA5_FORCING)
tgt_c_era5, src_c_era5, _ = run(masker_c_era5, ERA5_FORCING)
assert not np.any(mask_np(tgt_c_era5)), "Mode C ERA5: target must be all-False (forcing)"
assert np.all(mask_np(src_c_era5)),     "Mode C ERA5: source must be all-True (complement of all-False)"
print("  PASS ERA5: target=all-False (forcing), source=all-True")

# IASI: diagnostic=True → source=all-False, target stays all-True (rate=1.0).
# train_target_channels must be non-empty so is_stream_forcing returns False
# (otherwise the target would also be forced to all-False).
IASI_DIAGNOSTIC = {"name": "ERA5", "diagnostic": True, "train_target_channels": ["bt"]}
masker_c_iasi = make_masker(mode_c_cfg, IASI_DIAGNOSTIC)
tgt_c_iasi, src_c_iasi, _ = run(masker_c_iasi, IASI_DIAGNOSTIC)
assert not np.any(mask_np(src_c_iasi)), "Mode C IASI: source must be all-False (diagnostic)"
assert np.all(mask_np(tgt_c_iasi)),     "Mode C IASI: target must be all-True (rate=1.0)"
print("  PASS IASI: source=all-False (diagnostic), target=all-True (rate=1.0)")


# ── enabled:False filtering → falls through to Mode A ────────────────────────
print("=== enabled:False filtering: target_input with only disabled entries → Mode A ===")
mode_ef_cfg = OmegaConf.create({
    "model_input": {
        "mae_random": {
            "masking_strategy": "random",
            "num_samples": 1,
            "masking_strategy_config": {"rate": 0.15, "rate_sampling": False},
        },
    },
    "target_input": {
        # Only entry is disabled — should be filtered, triggering Mode A
        "disabled_target": {"masking_strategy": "random", "enabled": False, "num_samples": 0},
    },
    "losses": LOSSES_CFG,
})
masker_ef = make_masker(mode_ef_cfg, ERA5_STREAM)  # ERA5_STREAM has train_source_channels
tgt_ef, src_ef, _ = run(masker_ef, ERA5_STREAM)
src_ef_m = mask_np(src_ef)
tgt_ef_m = mask_np(tgt_ef)
assert not np.any(src_ef_m & tgt_ef_m), "enabled:False filter: Mode A → no overlap"
assert np.all(src_ef_m | tgt_ef_m),     "enabled:False filter: Mode A → full coverage"
print("  PASS: disabled target_input treated as auto-generated → Mode A complement override")


print("\nAll checks passed ✓")

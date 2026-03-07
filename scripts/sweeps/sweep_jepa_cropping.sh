#!/bin/bash
# =============================================================================
# JEPA Frozen Teacher - Cropping Masking Hyperparameter Sweep
# =============================================================================
#
# Launches N experiments with randomly sampled hyperparameters using
# cropping_healpix masking (geodesic_disk) with 2D RoPE.
#
# Usage:
#   bash sweep_jepa_cropping.sh <STRATEGY> [NUM_EXPERIMENTS]
#
#   STRATEGY (required):
#     contained      - Student disk contained within teacher (subset)
#     cone_distance  - Random angular separation 0-90° (variable overlap)
#     disjoint       - Fully separated student/teacher regions
#
#   NUM_EXPERIMENTS: default 10
#
# Examples:
#   bash sweep_jepa_cropping.sh contained 5
#   bash sweep_jepa_cropping.sh cone_distance 10
#   bash sweep_jepa_cropping.sh disjoint 8
#
# Swept parameters:
#   lr_max              log-uniform [1e-6, 5e-5]
#   student mask_rate   uniform     [0.1, 0.8]   (fraction of cells KEPT)
#
# Fixed design choices (not swept):
#   - Full encoder architecture from frozen sweep config
#   - JEPA loss with frozen teacher (p43hxwic), MLP head
#   - rope_2D: True
#   - Warmup 512 steps
#   - 32 mini-epochs, 4096 samples each
#
# =============================================================================

set -euo pipefail

# --- Args ---
STRATEGY="${1:?Usage: $0 <contained|cone_distance|disjoint> [NUM_EXPERIMENTS]}"
NUM_EXPERIMENTS="${2:-10}"

# --- Map strategy to config files ---
case "$STRATEGY" in
    contained)
        CONFIG="config_jepa_frozen_cropping_contained_2drope"
        FINETUNE_CONFIG="config_jepa_finetuning_cropping"
        ;;
    cone_distance)
        CONFIG="config_jepa_frozen_cropping_cone_distance_2drope"
        FINETUNE_CONFIG="config_jepa_finetuning_cropping_cone_distance"
        ;;
    disjoint)
        CONFIG="config_jepa_frozen_cropping_disjoint_2drope"
        FINETUNE_CONFIG="config_jepa_finetuning_cropping"
        ;;
    *)
        echo "ERROR: Unknown strategy '$STRATEGY'. Use: contained, cone_distance, or disjoint"
        exit 1
        ;;
esac

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# WeatherGenerator repo root (scripts/sweeps -> repo root)
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Parent dir containing both WeatherGenerator and WeatherGenerator-private
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/WeatherGenerator-private/hpc/launch-slurm-multi.py"
BASE_CONFIG="${REPO_ROOT}/config/${CONFIG}.yml"
FINETUNE_CONFIG_PATH="${REPO_ROOT}/config/${FINETUNE_CONFIG}.yml"
LOG_FILE="${SCRIPT_DIR}/sweep_cropping_${STRATEGY}_log.csv"

# --- Validate ---
[[ -f "$LAUNCHER" ]] || { echo "ERROR: launch-slurm-multi.py not found at $LAUNCHER"; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: base config not found at $BASE_CONFIG"; exit 1; }
[[ -f "$FINETUNE_CONFIG_PATH" ]] || { echo "ERROR: finetuning config not found at $FINETUNE_CONFIG_PATH"; exit 1; }

# --- CSV log header ---
echo "run_id,lr_max,mask_rate,strategy" > "$LOG_FILE"

# --- Sampling helper ---
sample_params() {
    python3 -c "
import random, math, string

run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

lr     = math.exp(random.uniform(math.log(1e-6), math.log(5e-5)))
mask   = random.uniform(0.1, 0.8)

print(f'{run_id},{lr:.2e},{mask:.2f}')
"
}

# --- Write per-experiment YAML overlay ---
write_exp_config() {
    local exp_num="$1"
    local lr="$2"
    local mask="$3"
    local out="${SCRIPT_DIR}/config_exp_${STRATEGY}_${exp_num}.yml"

    python3 -c "
import yaml

cfg = {
    '_base_': '${BASE_CONFIG}',
    'model_path': './models',
    'training_config': {
        'learning_rate_scheduling': {'lr_max': float('${lr}')},
        'model_input': {
            'student_cropping': {
                'masking_strategy_config': {'rate': float('${mask}')}
            }
        }
    },
    'wgtags': {'exp': 'jepa_cropping_${STRATEGY}_sweep'},
}
print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
" > "$out"

    echo "$out"
}

# --- Main ---
echo "========================================="
echo " JEPA Cropping Masking Sweep"
echo " Strategy    : $STRATEGY"
echo " Experiments : $NUM_EXPERIMENTS"
echo " Base config : $BASE_CONFIG"
echo " Finetune    : $FINETUNE_CONFIG_PATH"
echo " Log file    : $LOG_FILE"
echo "========================================="
echo ""

for i in $(seq 1 "$NUM_EXPERIMENTS"); do
    PARAMS=$(sample_params)
    IFS=',' read -r RUN_ID LR_MAX MASK_RATE <<< "$PARAMS"

    EXP_CONFIG=$(write_exp_config "$i" "$LR_MAX" "$MASK_RATE")

    echo "--- Experiment $i/$NUM_EXPERIMENTS [$RUN_ID] ---"
    echo "  lr_max         = $LR_MAX"
    echo "  mask_rate      = $MASK_RATE (fraction kept)"
    echo "  strategy       = $STRATEGY"
    echo "  config         = $EXP_CONFIG"

    echo "$RUN_ID,$LR_MAX,$MASK_RATE,$STRATEGY" >> "$LOG_FILE"

    "$LAUNCHER" \
        --run-id "$RUN_ID" \
        --chain-jobs 1 1 \
        --base-config "$BASE_CONFIG" \
        --config "$EXP_CONFIG" "$FINETUNE_CONFIG_PATH" \
        --nodes 1

    echo ""
done

echo "========================================="
echo " All $NUM_EXPERIMENTS experiments submitted"
echo " Strategy: $STRATEGY"
echo " Parameters logged to: $LOG_FILE"
echo "========================================="

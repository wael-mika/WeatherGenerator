#!/bin/bash
# =============================================================================
# JEPA Frozen Teacher - Cone Distance Cropping Sweep
# =============================================================================
#
# Launches N experiments with randomly sampled hyperparameters using
# cropping_healpix (geodesic_disk) with cone_distance relationship and 2D RoPE.
#
# Usage:
#   bash sweep_jepa_cropping_cone_distance.sh [NUM_EXPERIMENTS]
#
#   NUM_EXPERIMENTS: default 10
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
#   - cone_distance relationship (0-90° random angular separation)
#
# =============================================================================

set -euo pipefail

# --- Logging ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$SWEEP_LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SWEEP_LOG_DIR}/sweep_cone_distance_${TIMESTAMP}.log"
}

log "=== Sweep script started ==="
log "Working directory: $(pwd)"
log "Script: ${BASH_SOURCE[0]}"
log "Args: $*"
log "Python3: $(which python3 2>&1 || echo 'NOT FOUND')"
log "Python3 version: $(python3 --version 2>&1 || echo 'FAILED')"

# --- Args ---
NUM_EXPERIMENTS="${1:-10}"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/WeatherGenerator-private/hpc/launch-slurm-multi.py"
BASE_CONFIG="${REPO_ROOT}/config/config_jepa_frozen_cropping_cone_distance_2drope.yml"
FINETUNE_CONFIG="${REPO_ROOT}/config/config_jepa_finetuning_cropping_cone_distance.yml"
LOG_FILE="${SCRIPT_DIR}/sweep_cone_distance_log.csv"

log "SCRIPT_DIR: $SCRIPT_DIR"
log "REPO_ROOT: $REPO_ROOT"
log "PROJECT_ROOT: $PROJECT_ROOT"
log "LAUNCHER: $LAUNCHER"
log "BASE_CONFIG: $BASE_CONFIG"
log "FINETUNE_CONFIG: $FINETUNE_CONFIG"

# --- Validate ---
if [[ ! -f "$LAUNCHER" ]]; then
    log "ERROR: launch-slurm-multi.py not found at $LAUNCHER"
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    log "ERROR: base config not found at $BASE_CONFIG"
    exit 1
fi
if [[ ! -f "$FINETUNE_CONFIG" ]]; then
    log "ERROR: finetuning config not found at $FINETUNE_CONFIG"
    exit 1
fi
log "All files validated OK"

# --- CSV log header ---
echo "run_id,lr_max,mask_rate" > "$LOG_FILE"

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
    local out="${SCRIPT_DIR}/config_exp_cone_distance_${exp_num}.yml"

    cat > "$out" << YAML_EOF
# Auto-generated overlay for cone_distance experiment ${exp_num}
training_config:
  learning_rate_scheduling:
    lr_max: ${lr}
  model_input:
    student_cropping:
      masking_strategy_config:
        rate: ${mask}
wgtags:
  exp: "jepa_cropping_cone_distance_sweep"
YAML_EOF

    log "Generated overlay config at $out" >&2
    echo "$out"
}

# --- Main ---
log "========================================="
log " JEPA Cropping Cone Distance Sweep"
log " Experiments : $NUM_EXPERIMENTS"
log " Base config : $BASE_CONFIG"
log " Finetune    : $FINETUNE_CONFIG"
log " Log file    : $LOG_FILE"
log "========================================="
echo ""

for i in $(seq 1 "$NUM_EXPERIMENTS"); do
    PARAMS=$(sample_params)
    IFS=',' read -r RUN_ID LR_MAX MASK_RATE <<< "$PARAMS"

    EXP_CONFIG=$(write_exp_config "$i" "$LR_MAX" "$MASK_RATE")

    log "--- Experiment $i/$NUM_EXPERIMENTS [$RUN_ID] ---"
    log "  lr_max         = $LR_MAX"
    log "  mask_rate      = $MASK_RATE (fraction kept)"
    log "  config         = $EXP_CONFIG"

    if [[ ! -f "$EXP_CONFIG" ]]; then
        log "ERROR: Generated config file not found at $EXP_CONFIG"
        continue
    fi
    log "  config size    = $(wc -c < "$EXP_CONFIG") bytes"
    log "  config content:"
    cat "$EXP_CONFIG" >> "${SWEEP_LOG_DIR}/sweep_cone_distance_${TIMESTAMP}.log"

    echo "$RUN_ID,$LR_MAX,$MASK_RATE" >> "$LOG_FILE"

    log "Launching: $LAUNCHER --run-id $RUN_ID --chain-jobs 1 1 --base-config $BASE_CONFIG --config $EXP_CONFIG $FINETUNE_CONFIG --nodes 1"

    if ! "$LAUNCHER" \
        --run-id "$RUN_ID" \
        --chain-jobs 1 1 \
        --base-config "$BASE_CONFIG" \
        --config "$EXP_CONFIG" "$FINETUNE_CONFIG" \
        --nodes 1 2>&1 | tee -a "${SWEEP_LOG_DIR}/sweep_cone_distance_${TIMESTAMP}.log"; then
        log "ERROR: Launcher failed for experiment $i [$RUN_ID] (exit code: ${PIPESTATUS[0]})"
        log "Continuing to next experiment..."
    else
        log "Experiment $i [$RUN_ID] submitted successfully"
    fi

    echo ""
done

log "========================================="
log " All $NUM_EXPERIMENTS experiments submitted"
log " Parameters logged to: $LOG_FILE"
log " Debug log: ${SWEEP_LOG_DIR}/sweep_cone_distance_${TIMESTAMP}.log"
log "========================================="

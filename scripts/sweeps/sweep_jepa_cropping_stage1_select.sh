#!/bin/bash
# =============================================================================
# JEPA Frozen Teacher - Random Student Masking Sweep
# =============================================================================
#
# Launches N experiments with randomly sampled hyperparameters for the ablation
# setup where the teacher always sees the full field.
#
# Usage:
#   bash sweep_jepa_cropping_stage1_select.sh [NUM_EXPERIMENTS] [--stage1-config PATH]
#   bash sweep_jepa_cropping_stage1_select.sh --stage1-config PATH [NUM_EXPERIMENTS]
#
#   NUM_EXPERIMENTS: default 10
#   --stage1-config PATH:
#       Optional Stage 1 base config to pass to launch-slurm-multi.py
#       Default: config/config_jepa_frozen_q200_2drope_qkrms.yml
#
# Swept parameters:
#   lr_max              log-uniform [1e-6, 5e-5]
#   student mask_rate   uniform     [0.10, 0.90]   (fraction of cells kept)
#
# Fixed design choices:
#   - Stage 2 config: config/config_jepa_finetuning.yml
#   - teacher mask_rate: 1.0 (full field)
#
# =============================================================================

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash sweep_jepa_cropping_stage1_select.sh [NUM_EXPERIMENTS] [--stage1-config PATH]
  bash sweep_jepa_cropping_stage1_select.sh --stage1-config PATH [NUM_EXPERIMENTS]

Options:
  --stage1-config PATH   Stage 1 base config passed to launch-slurm-multi.py
                         Default: config/config_jepa_frozen_q200_2drope_qkrms.yml
  -h, --help             Show this help message
EOF
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SWEEP_NAME="jepa_random_student_rate"
SWEEP_LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/logs"
mkdir -p "$SWEEP_LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${SWEEP_LOG_DIR}/sweep_${TIMESTAMP}.log"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
LAUNCHER="${PROJECT_ROOT}/WeatherGenerator-private/hpc/launch-slurm-multi.py"
DEFAULT_STAGE1_CONFIG="config/config_jepa_frozen_q200_2drope_qkrms.yml"
STAGE1_CONFIG_INPUT="$DEFAULT_STAGE1_CONFIG"
FINETUNE_CONFIG_PATH="${REPO_ROOT}/config/config_jepa_finetuning.yml"
CONFIG_RUNS_DIR="${REPO_ROOT}/config/sweep_runs"
LOG_FILE_STEM="sweep_jepa_random_student_rate_stage1_select_log_${TIMESTAMP}"

NUM_EXPERIMENTS="10"
NUM_EXPERIMENTS_SET=0
ORIGINAL_ARGS=("$@")

resolve_file_path() {
    local input="$1"
    local candidate
    local resolved_dir

    for candidate in "$input" "${REPO_ROOT}/${input}"; do
        if [[ -f "$candidate" ]]; then
            resolved_dir="$(cd "$(dirname "$candidate")" >/dev/null 2>&1 && pwd)"
            printf '%s/%s\n' "$resolved_dir" "$(basename "$candidate")"
            return 0
        fi
    done

    return 1
}

launcher_config_arg() {
    local path="$1"

    if [[ "$path" == "$REPO_ROOT/"* ]]; then
        printf './%s\n' "${path#"$REPO_ROOT/"}"
    else
        printf '%s\n' "$path"
    fi
}

resolve_unique_log_file() {
    local candidate="${SCRIPT_DIR}/${LOG_FILE_STEM}.csv"
    local suffix=1

    while [[ -e "$candidate" ]]; do
        candidate="${SCRIPT_DIR}/${LOG_FILE_STEM}_$(printf '%02d' "$suffix").csv"
        suffix=$((suffix + 1))
    done

    printf '%s\n' "$candidate"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage1-config)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --stage1-config requires a path argument" >&2
                usage >&2
                exit 1
            fi
            STAGE1_CONFIG_INPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ "$NUM_EXPERIMENTS_SET" -eq 1 ]]; then
                echo "ERROR: Unexpected extra positional argument '$1'" >&2
                usage >&2
                exit 1
            fi
            NUM_EXPERIMENTS="$1"
            NUM_EXPERIMENTS_SET=1
            shift
            ;;
    esac
done

if ! [[ "$NUM_EXPERIMENTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_EXPERIMENTS must be a positive integer, got '$NUM_EXPERIMENTS'" >&2
    exit 1
fi

if ! BASE_CONFIG=$(resolve_file_path "$STAGE1_CONFIG_INPUT"); then
    echo "ERROR: Stage 1 config not found at '$STAGE1_CONFIG_INPUT'" >&2
    exit 1
fi
BASE_CONFIG_LAUNCH_ARG="$(launcher_config_arg "$BASE_CONFIG")"
FINETUNE_CONFIG_LAUNCH_ARG="$(launcher_config_arg "$FINETUNE_CONFIG_PATH")"
LOG_FILE="$(resolve_unique_log_file)"

mkdir -p "$CONFIG_RUNS_DIR"
export UV_CACHE_DIR="${HOME}/.cache/uv"
mkdir -p "$UV_CACHE_DIR"

log "=== Sweep script started ==="
log "Working directory: $(pwd)"
log "Script: ${BASH_SOURCE[0]}"
log "Args: ${ORIGINAL_ARGS[*]}"
log "Python3: $(which python3 2>&1 || echo 'NOT FOUND')"
log "Python3 version: $(python3 --version 2>&1 || echo 'FAILED')"

LR_MIN="1e-6"
LR_MAX_CAP="5e-5"
STUDENT_MASK_MIN="0.10"
STUDENT_MASK_MAX="0.90"
TEACHER_MASK_RATE="1.0"

log "SWEEP_NAME: $SWEEP_NAME"
log "SCRIPT_DIR: $SCRIPT_DIR"
log "REPO_ROOT: $REPO_ROOT"
log "PROJECT_ROOT: $PROJECT_ROOT"
log "LAUNCHER: $LAUNCHER"
log "BASE_CONFIG: $BASE_CONFIG"
log "FINETUNE_CONFIG_PATH: $FINETUNE_CONFIG_PATH"
log "CONFIG_RUNS_DIR: $CONFIG_RUNS_DIR"
log "UV_CACHE_DIR: $UV_CACHE_DIR"
log "LR_RANGE: [$LR_MIN, $LR_MAX_CAP]"
log "STUDENT_MASK_RANGE: [$STUDENT_MASK_MIN, $STUDENT_MASK_MAX]"
log "TEACHER_MASK_RATE: $TEACHER_MASK_RATE"

if [[ ! -f "$LAUNCHER" ]]; then
    log "ERROR: launch-slurm-multi.py not found at $LAUNCHER"
    log "Contents of $PROJECT_ROOT: $(ls "$PROJECT_ROOT" 2>&1 || echo 'DIR NOT FOUND')"
    exit 1
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
    log "ERROR: base config not found at $BASE_CONFIG"
    exit 1
fi
if [[ ! -f "$FINETUNE_CONFIG_PATH" ]]; then
    log "ERROR: finetuning config not found at $FINETUNE_CONFIG_PATH"
    exit 1
fi
log "All files validated OK"

echo "run_id,lr_max,student_mask_rate,teacher_mask_rate,sweep_name,stage1_config" > "$LOG_FILE"

sample_params() {
    LR_MIN="$LR_MIN" \
    LR_MAX_CAP="$LR_MAX_CAP" \
    STUDENT_MASK_MIN="$STUDENT_MASK_MIN" \
    STUDENT_MASK_MAX="$STUDENT_MASK_MAX" \
    python3 - <<'PY'
import math
import os
import random
import string

lr_min = float(os.environ["LR_MIN"])
lr_max = float(os.environ["LR_MAX_CAP"])
student_min = float(os.environ["STUDENT_MASK_MIN"])
student_max = float(os.environ["STUDENT_MASK_MAX"])

run_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
lr = math.exp(random.uniform(math.log(lr_min), math.log(lr_max)))
student = float(f"{random.uniform(student_min, student_max):.2f}")
print(f"{run_id},{lr:.2e},{student:.2f}")
PY
}

write_exp_config() {
    local exp_num="$1"
    local run_id="$2"
    local lr="$3"
    local student_mask="$4"
    local out="${CONFIG_RUNS_DIR}/pretrain_random_student_${TIMESTAMP}_${exp_num}_${run_id}.yml"

    cat > "$out" <<YAML_EOF
# Auto-generated overlay for experiment ${exp_num}
training_config:
  learning_rate_scheduling:
    lr_max: ${lr}
  model_input:
    random_easy:
      masking_strategy_config:
        rate: ${student_mask}
  target_input:
    random_easy_target:
      masking_strategy_config:
        rate: ${TEACHER_MASK_RATE}
wgtags:
  exp: "${SWEEP_NAME}"
YAML_EOF

    log "Generated overlay config at $out" >&2
    echo "$out"
}

log "========================================="
log " JEPA Random Student Masking Sweep"
log " Experiments : $NUM_EXPERIMENTS"
log " Stage 1 cfg : $BASE_CONFIG"
log " Finetune    : $FINETUNE_CONFIG_PATH"
log " Log file    : $LOG_FILE"
log " Debug log   : ${SWEEP_LOG_DIR}/sweep_${TIMESTAMP}.log"
log "========================================="
echo ""

for i in $(seq 1 "$NUM_EXPERIMENTS"); do
    PARAMS=$(sample_params)
    IFS=',' read -r RUN_ID LR_MAX STUDENT_MASK_RATE <<< "$PARAMS"

    EXP_CONFIG=$(write_exp_config "$i" "$RUN_ID" "$LR_MAX" "$STUDENT_MASK_RATE")
    EXP_CONFIG_LAUNCH_ARG="$(launcher_config_arg "$EXP_CONFIG")"

    log "--- Experiment $i/$NUM_EXPERIMENTS [$RUN_ID] ---"
    log "  lr_max         = $LR_MAX"
    log "  student_rate   = $STUDENT_MASK_RATE (fraction kept)"
    log "  teacher_rate   = $TEACHER_MASK_RATE (full field)"
    log "  stage1 cfg     = $BASE_CONFIG_LAUNCH_ARG"
    log "  pretrain ovl   = $EXP_CONFIG_LAUNCH_ARG"
    log "  finetune cfg   = $FINETUNE_CONFIG_LAUNCH_ARG"

    if [[ ! -f "$EXP_CONFIG" ]]; then
        log "ERROR: Generated config file not found at $EXP_CONFIG"
        continue
    fi
    log "  config size    = $(wc -c < "$EXP_CONFIG") bytes"
    log "  config content:"
    cat "$EXP_CONFIG" >> "${SWEEP_LOG_DIR}/sweep_${TIMESTAMP}.log"

    echo "$RUN_ID,$LR_MAX,$STUDENT_MASK_RATE,$TEACHER_MASK_RATE,$SWEEP_NAME,$BASE_CONFIG_LAUNCH_ARG" >> "$LOG_FILE"

    log "Launching: $LAUNCHER --run-id $RUN_ID --chain-jobs 1 1 --base-config $BASE_CONFIG_LAUNCH_ARG --config $EXP_CONFIG_LAUNCH_ARG $FINETUNE_CONFIG_LAUNCH_ARG --nodes 1"

    if ! "$LAUNCHER" \
        --run-id "$RUN_ID" \
        --chain-jobs 1 1 \
        --base-config "$BASE_CONFIG_LAUNCH_ARG" \
        --config "$EXP_CONFIG_LAUNCH_ARG" "$FINETUNE_CONFIG_LAUNCH_ARG" \
        --nodes 1 2>&1 | tee -a "${SWEEP_LOG_DIR}/sweep_${TIMESTAMP}.log"; then
        log "ERROR: Launcher failed for experiment $i [$RUN_ID] (exit code: ${PIPESTATUS[0]})"
        log "Continuing to next experiment..."
    else
        log "Experiment $i [$RUN_ID] submitted successfully"
        log "Expected stage outputs under: output/${RUN_ID}-stage1 and output/${RUN_ID}-stage2"
    fi

    echo ""
done

log "========================================="
log " All $NUM_EXPERIMENTS experiments submitted"
log " Parameters logged to: $LOG_FILE"
log " Debug log: ${SWEEP_LOG_DIR}/sweep_${TIMESTAMP}.log"
log "========================================="

#!/bin/bash
# =============================================================================
# JEPA Frozen Teacher - Random Student Masking Sweep
# =============================================================================
#
# Launches N experiments with randomly sampled hyperparameters for the ablation
# setup where the teacher always sees the full field.
#
# Usage:
#   bash sweep_jepa_cropping_stage1_select_qzt.sh [NUM_EXPERIMENTS] [--stage1-config PATH] [-q] [-z] [-t]
#   bash sweep_jepa_cropping_stage1_select_qzt.sh --stage1-config PATH [NUM_EXPERIMENTS] [-q] [-z] [-t]
#
#   NUM_EXPERIMENTS: default 10
#   --stage1-config PATH:
#       Optional Stage 1 base config to pass to launch-slurm-multi.py
#       Default: config/config_jepa_frozen_mtm_sweep_2drope.yml
#   -q / -z / -t:
#       Before launching, edit config/streams/era5_1deg_abl/era5.yml in place
#       so source and target contain only the selected pressure-level family or families.
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
  bash sweep_jepa_cropping_stage1_select_qzt.sh [NUM_EXPERIMENTS] [--stage1-config PATH] [-q] [-z] [-t]
  bash sweep_jepa_cropping_stage1_select_qzt.sh --stage1-config PATH [NUM_EXPERIMENTS] [-q] [-z] [-t]

Options:
  --stage1-config PATH   Stage 1 base config passed to launch-slurm-multi.py
                         Default: config/config_jepa_frozen_mtm_sweep_2drope.yml
  -q                     Edit config/streams/era5_1deg_abl/era5.yml to use only q_* levels
  -z                     Edit config/streams/era5_1deg_abl/era5.yml to use only z_* levels
  -t                     Edit config/streams/era5_1deg_abl/era5.yml to use only t_* levels
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
DEFAULT_STAGE1_CONFIG="config/config_jepa_frozen_mtm_sweep_2drope.yml"
STAGE1_CONFIG_INPUT="$DEFAULT_STAGE1_CONFIG"
FINETUNE_CONFIG_PATH="${REPO_ROOT}/config/config_jepa_finetuning.yml"
CONFIG_RUNS_DIR="${REPO_ROOT}/config/sweep_runs"
FIXED_STREAMS_DIR_REL="config/streams/era5_1deg_abl"
FIXED_STREAMS_DIR_LAUNCH_ARG="./config/streams/era5_1deg_abl/"
FIXED_STREAM_FILE="${REPO_ROOT}/${FIXED_STREAMS_DIR_REL}/era5.yml"
LOG_FILE_STEM="sweep_jepa_random_student_rate_stage1_select_qzt_log_${TIMESTAMP}"

NUM_EXPERIMENTS="10"
NUM_EXPERIMENTS_SET=0
ORIGINAL_ARGS=("$@")
SELECT_Q=0
SELECT_Z=0
SELECT_T=0
STREAM_SELECTION_TAG="default"
STREAM_OVERRIDE_DIR=""
STREAM_OVERRIDE_LAUNCH_ARG=""
SELECTED_STREAM_GROUPS=()
SELECTED_STREAM_VARS=()

Q_STREAM_VARS=(
    q_1000 q_925 q_850 q_700 q_600
    q_500 q_400 q_300 q_250 q_200
)
Z_STREAM_VARS=(
    z_1000 z_925 z_850 z_700 z_600
    z_500 z_400 z_300 z_250 z_200
)
T_STREAM_VARS=(
    t_1000 t_925 t_850 t_700 t_600
    t_500 t_400 t_300 t_250 t_200
)

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

generate_stream_overlay_file() {
    local source_stream_file="$1"
    local output_stream_file="$2"
    shift 2

    STREAM_SOURCE_FILE="$source_stream_file" \
    STREAM_OUTPUT_FILE="$output_stream_file" \
    python3 - "$@" <<'PY'
from pathlib import Path
import os
import re
import sys

source_file = Path(os.environ["STREAM_SOURCE_FILE"])
output_file = Path(os.environ["STREAM_OUTPUT_FILE"])
variables = sys.argv[1:]

if not variables:
    raise SystemExit("No stream variables selected")

lines = source_file.read_text().splitlines()

def make_block_lines(key):
    block = [f"  {key} : ["]
    for index, variable in enumerate(variables):
        suffix = "," if index < len(variables) - 1 else ""
        block.append(f"    '{variable}'{suffix}")
    block.append("  ]")
    return block

def replace_or_insert_block(current_lines, key, preferred_before=None, fallback_after=None):
    block_lines = make_block_lines(key)
    key_pattern = re.compile(rf"^\s{{2}}{re.escape(key)}\s*:")

    start = next(
        (index for index, line in enumerate(current_lines) if key_pattern.match(line)),
        None,
    )
    if start is not None:
        end = next(
            (index for index in range(start, len(current_lines)) if current_lines[index].strip() == "]"),
            None,
        )
        if end is None:
            raise SystemExit(f"Could not find end of {key} block in {source_file}")
        return current_lines[:start] + block_lines + current_lines[end + 1 :]

    insert_at = None
    if preferred_before is not None:
        preferred_pattern = re.compile(rf"^\s{{2}}{re.escape(preferred_before)}\s*:")
        insert_at = next(
            (index for index, line in enumerate(current_lines) if preferred_pattern.match(line)),
            None,
        )

    if insert_at is None and fallback_after is not None:
        fallback_pattern = re.compile(rf"^\s{{2}}{re.escape(fallback_after)}\s*:")
        after_index = next(
            (index for index, line in enumerate(current_lines) if fallback_pattern.match(line)),
            None,
        )
        if after_index is not None:
            insert_at = after_index + 1

    if insert_at is None:
        raise SystemExit(f"Could not place {key} block in {source_file}")

    return current_lines[:insert_at] + block_lines + current_lines[insert_at:]

lines = replace_or_insert_block(
    lines,
    "source",
    preferred_before="source_exclude",
    fallback_after="stream_id",
)
lines = replace_or_insert_block(
    lines,
    "target",
    preferred_before="target_exclude",
    fallback_after="source_exclude",
)

output_file.write_text("\n".join(lines) + "\n")
PY
}

prepare_stream_override() {
    if [[ ${#SELECTED_STREAM_VARS[@]} -eq 0 ]]; then
        return 0
    fi

    if [[ ! -f "$FIXED_STREAM_FILE" ]]; then
        echo "ERROR: Expected era5.yml at '$FIXED_STREAM_FILE'" >&2
        exit 1
    fi

    generate_stream_overlay_file \
        "$FIXED_STREAM_FILE" \
        "$FIXED_STREAM_FILE" \
        "${SELECTED_STREAM_VARS[@]}"

    STREAM_OVERRIDE_DIR="${REPO_ROOT}/${FIXED_STREAMS_DIR_REL}"
    STREAM_OVERRIDE_LAUNCH_ARG="${FIXED_STREAMS_DIR_LAUNCH_ARG}"
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
        -q)
            SELECT_Q=1
            shift
            ;;
        -z)
            SELECT_Z=1
            shift
            ;;
        -t)
            SELECT_T=1
            shift
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

if [[ "$SELECT_Q" -eq 1 ]]; then
    SELECTED_STREAM_GROUPS+=("q")
    SELECTED_STREAM_VARS+=("${Q_STREAM_VARS[@]}")
fi
if [[ "$SELECT_Z" -eq 1 ]]; then
    SELECTED_STREAM_GROUPS+=("z")
    SELECTED_STREAM_VARS+=("${Z_STREAM_VARS[@]}")
fi
if [[ "$SELECT_T" -eq 1 ]]; then
    SELECTED_STREAM_GROUPS+=("t")
    SELECTED_STREAM_VARS+=("${T_STREAM_VARS[@]}")
fi
if [[ ${#SELECTED_STREAM_GROUPS[@]} -gt 0 ]]; then
    STREAM_SELECTION_TAG="$(IFS=_; echo "${SELECTED_STREAM_GROUPS[*]}")"
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
prepare_stream_override

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
log "FIXED_STREAM_FILE: $FIXED_STREAM_FILE"
log "UV_CACHE_DIR: $UV_CACHE_DIR"
log "LR_RANGE: [$LR_MIN, $LR_MAX_CAP]"
log "STUDENT_MASK_RANGE: [$STUDENT_MASK_MIN, $STUDENT_MASK_MAX]"
log "TEACHER_MASK_RATE: $TEACHER_MASK_RATE"
log "STREAM_SELECTION: $STREAM_SELECTION_TAG"
if [[ -n "$STREAM_OVERRIDE_DIR" ]]; then
    log "STREAM_OVERRIDE_DIR: $STREAM_OVERRIDE_DIR"
fi

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
    local streams_directory_line=""

    if [[ -n "$STREAM_OVERRIDE_LAUNCH_ARG" ]]; then
        streams_directory_line="streams_directory: \"${STREAM_OVERRIDE_LAUNCH_ARG}\""
    fi

    cat > "$out" <<YAML_EOF
# Auto-generated overlay for experiment ${exp_num}
${streams_directory_line}
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
  stream_family: "${STREAM_SELECTION_TAG}"
YAML_EOF

    log "Generated overlay config at $out" >&2
    echo "$out"
}

log "========================================="
log " JEPA Random Student Masking Sweep"
log " Experiments : $NUM_EXPERIMENTS"
log " Stage 1 cfg : $BASE_CONFIG"
log " Finetune    : $FINETUNE_CONFIG_PATH"
log " Stream vars : $STREAM_SELECTION_TAG"
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
    if [[ -n "$STREAM_OVERRIDE_LAUNCH_ARG" ]]; then
        log "  stream ovl dir = $STREAM_OVERRIDE_LAUNCH_ARG"
    fi

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

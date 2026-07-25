#!/usr/bin/env bash
set -euo pipefail

# Auto train/eval loop for the existing oracle LeRobot dataset.
# Stops when a 100-episode oracle eval reaches TARGET_SUCCESS.

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
PY="conda run -n ${CONDA_ENV} --no-capture-output python"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/dataset_oracle_lerobot}"
REPO_ID="${REPO_ID:-local/avla_select_radio_oracle}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
TARGET_SUCCESS="${TARGET_SUCCESS:-0.60}"

TRAIN_STEPS="${TRAIN_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_EVERY="${SAVE_EVERY:-4000}"
LOG_EVERY="${LOG_EVERY:-20}"

EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"

NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"

CURRENT_OUTPUT="${CURRENT_OUTPUT:-${REPO_ROOT}/outputs/smolvla_oracle_class_tokens}"
CURRENT_EVAL="${CURRENT_EVAL:-${REPO_ROOT}/outputs/eval_smolvla_oracle_class_tokens}"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/outputs/auto_loop_logs"
MASTER_LOG="${MASTER_LOG:-${REPO_ROOT}/outputs/auto_loop_logs/loop.log}"

log() {
    printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${MASTER_LOG}"
}

success_rate() {
    local summary="$1"
    if [[ ! -f "${summary}" ]]; then
        echo "nan"
        return
    fi
    ${PY} - "${summary}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(float(json.load(f).get("success_rate", float("nan"))))
PY
}

rate_ge_target() {
    local rate="$1"
    ${PY} - "${rate}" "${TARGET_SUCCESS}" <<'PY'
import math, sys
rate = float(sys.argv[1])
target = float(sys.argv[2])
sys.exit(0 if math.isfinite(rate) and rate >= target else 1)
PY
}

wait_for_current_run() {
    local summary="${CURRENT_EVAL}/eval_summary.json"
    if [[ -f "${summary}" ]]; then
        return
    fi
    if pgrep -af "smolvla_oracle_class_tokens|eval_smolvla_oracle_class_tokens" >/dev/null; then
        log "waiting for current class_tokens run to finish"
        while pgrep -af "smolvla_oracle_class_tokens|eval_smolvla_oracle_class_tokens" >/dev/null; do
            sleep 300
            log "still waiting for current run"
        done
    fi
}

eval_ckpt() {
    local ckpt="$1"
    local eval_dir="$2"
    mkdir -p "${eval_dir}"
    log "eval ${ckpt} -> ${eval_dir}"
    ${PY} src/eval/eval_smolvla_audio.py \
        --ckpt "${ckpt}" \
        --pretrained "${PRETRAINED}" \
        --taxonomy "${TAXONOMY}" \
        --task-name select_radio \
        --n-episodes "${EVAL_N}" \
        --max-episode-length "${EVAL_MAX_LEN}" \
        --horizon "${EVAL_HORIZON}" \
        --save-dir "${eval_dir}" \
        --oracle-mode \
        --noise-az-std "${NOISE_AZ_STD}" \
        --noise-el-std "${NOISE_EL_STD}" \
        --noise-conf-min "${NOISE_CONF_MIN}" \
        --noise-conf-max "${NOISE_CONF_MAX}" \
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
        2>&1 | tee "${eval_dir}/eval.log"
}

train_variant() {
    local name="$1"
    shift
    local out="${REPO_ROOT}/outputs/${name}"
    local eval_dir="${REPO_ROOT}/outputs/eval_${name}"
    local log_file="${REPO_ROOT}/outputs/auto_loop_logs/${name}.train.log"

    rm -rf "${out}" "${eval_dir}"
    mkdir -p "${out}" "${eval_dir}"
    log "train ${name}"
    ${PY} src/training/train_smolvla_audio.py \
        --dataset-root "${DATASET_ROOT}" \
        --repo-id "${REPO_ID}" \
        --pretrained "${PRETRAINED}" \
        --taxonomy "${TAXONOMY}" \
        --output-dir "${out}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${TRAIN_STEPS}" \
        --lr 5e-5 \
        --audio-lr 1e-4 \
        --warmup-steps 500 \
        --num-workers "${NUM_WORKERS}" \
        --save-every "${SAVE_EVERY}" \
        --log-every "${LOG_EVERY}" \
        --oracle-noise on \
        --noise-az-std "${NOISE_AZ_STD}" \
        --noise-el-std "${NOISE_EL_STD}" \
        --noise-conf-min "${NOISE_CONF_MIN}" \
        --noise-conf-max "${NOISE_CONF_MAX}" \
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
        "$@" \
        2>&1 | tee "${log_file}"

    local ckpt
    ckpt="$(ls -t "${out}"/ckpt_step*.pt | head -1)"
    eval_ckpt "${ckpt}" "${eval_dir}"
    local rate
    rate="$(success_rate "${eval_dir}/eval_summary.json")"
    log "${name} success_rate=${rate}"
    if rate_ge_target "${rate}"; then
        log "target reached by ${name}"
        exit 0
    fi
}

log "auto loop start target=${TARGET_SUCCESS}"
wait_for_current_run

current_rate="$(success_rate "${CURRENT_EVAL}/eval_summary.json")"
log "current class_tokens success_rate=${current_rate}"
if rate_ge_target "${current_rate}"; then
    log "target reached by current class_tokens run"
    exit 0
fi

# Variants are ordered from conservative to more invasive. All use the existing
# dataset and 100-episode oracle eval.
train_variant smolvla_oracle_class_tokens_lm4 \
    --audio-fusion-mode class_tokens \
    --class-token-scale 0.1 \
    --unfreeze-last-n-lm-layers 4

train_variant smolvla_oracle_class_tokens_lora_full \
    --audio-fusion-mode class_tokens \
    --class-token-scale 0.1 \
    --unfreeze-last-n-lm-layers 0 \
    --vlm-lora \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

train_variant smolvla_oracle_inline_lm4_lowflip \
    --audio-fusion-mode inline \
    --unfreeze-last-n-lm-layers 4 \
    --noise-class-flip-prob 0.0

log "all configured variants failed to reach target=${TARGET_SUCCESS}"
exit 2

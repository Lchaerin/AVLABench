#!/usr/bin/env bash
set -euo pipefail

LEVEL="${LEVEL:?LEVEL is required}"
AZ_STD="${AZ_STD:?AZ_STD is required}"
EL_STD="${EL_STD:?EL_STD is required}"
CLASS_FLIP="${CLASS_FLIP:?CLASS_FLIP is required}"
SOURCE_DROP="${SOURCE_DROP:?SOURCE_DROP is required}"
DISTRACTOR_PROB="${DISTRACTOR_PROB:?DISTRACTOR_PROB is required}"
DISTRACTOR_CONF_MAX="${DISTRACTOR_CONF_MAX:?DISTRACTOR_CONF_MAX is required}"

REPO_ROOT="/home/rllab/Desktop/AVLABench"
cd "${REPO_ROOT}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

PY="conda run -n vlabench --no-capture-output python"
CKPT="${CKPT:-outputs/smolvla_oracle_all_mixed_fixed_one_nl_azcanon_lora_80k_v2/ckpt_step0080000.pt}"
PRETRAINED="${PRETRAINED:-/home/rllab/.cache/huggingface/hub/models--lerobot--smolvla_vlabench/snapshots/4fd586e12dc14b04d9d606ddbb77448df4f0ff29}"
OUT_ROOT="${OUT_ROOT:-outputs/noise_sweep_smolvla_oracle_all_mixed_fixed_one_nl_azcanon_lora_80k_v2_orig}"
EVAL_N="${EVAL_N:-100}"

run_task() {
    local task_name="$1"
    local save_dir="${OUT_ROOT}/level_${LEVEL}_${task_name}"
    echo "============================================================"
    echo "[noise level ${LEVEL}] task=${task_name}"
    echo "  az=${AZ_STD} el=${EL_STD} flip=${CLASS_FLIP}"
    echo "  drop=${SOURCE_DROP} distractor=${DISTRACTOR_PROB} distractor_conf_max=${DISTRACTOR_CONF_MAX}"
    echo "  out=${save_dir}"
    echo "============================================================"
    ${PY} src/eval/eval_smolvla_audio.py \
        --ckpt "${CKPT}" \
        --pretrained "${PRETRAINED}" \
        --taxonomy class_taxonomy.yaml \
        --task-name "${task_name}" \
        --n-episodes "${EVAL_N}" \
        --max-episode-length 200 \
        --horizon 5 \
        --intention-threshold 0.1 \
        --intention-mode exclusive \
        --exclusive-intention-margin 0.03 \
        --save-dir "${save_dir}" \
        --oracle-mode \
        --noise-az-std "${AZ_STD}" \
        --noise-el-std "${EL_STD}" \
        --noise-conf-min 0.85 \
        --noise-conf-max 0.98 \
        --noise-class-flip-prob "${CLASS_FLIP}" \
        --noise-source-drop-prob "${SOURCE_DROP}" \
        --noise-distractor-prob "${DISTRACTOR_PROB}" \
        --noise-distractor-conf-max "${DISTRACTOR_CONF_MAX}" \
        --class-smoothing-window 5 \
        --shuffle-audio-slots on \
        --target-first-audio-slots off \
        --canonicalize-audio-slots azimuth
}

run_task select_radio
run_task select_radio_two
run_task select_radio_silent

#!/usr/bin/env bash
# Train one SmolVLA policy on both two-radio instruction types:
#   1) select_radio_two: press the radio playing the requested sound class.
#   2) select_radio_silent: press the quiet radio while the other two radios sound.
#
# The source HDF5 datasets are combined into one temporary task directory and
# converted once with per-episode instructions preserved. Evaluation runs the
# same checkpoint on both tasks unless EVAL_TASKS is narrowed.

set -euo pipefail

REPO_ROOT="/home/rllab/Desktop/AVLABench"

TWO_SRC_DIR="${TWO_SRC_DIR:-${REPO_ROOT}/dataset_oracle_two/select_radio_two}"
SILENT_SRC_DIR="${SILENT_SRC_DIR:-${REPO_ROOT}/dataset_oracle_two_silent/select_radio_silent}"
COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${REPO_ROOT}/dataset_oracle_two_mixed}"
COMBINED_TASK_NAME="${COMBINED_TASK_NAME:-select_radio_two_mixed}"
COMBINED_SRC_DIR="${COMBINED_DATASET_DIR}/${COMBINED_TASK_NAME}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_oracle_two_mixed_lerobot}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_oracle_two_mixed_nl_azcanon_lora_40k_v1}"
EVAL_DIR_TWO="${EVAL_DIR_TWO:-${REPO_ROOT}/outputs/eval_smolvla_oracle_two_mixed_nl_azcanon_lora_40k_v1_select_radio_two}"
EVAL_DIR_SILENT="${EVAL_DIR_SILENT:-${REPO_ROOT}/outputs/eval_smolvla_oracle_two_mixed_nl_azcanon_lora_40k_v1_select_radio_silent}"

TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
REPO_ID="${REPO_ID:-local/avla_select_radio_two_mixed_oracle}"

DO_COMBINE="${DO_COMBINE:-1}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

TOP_K="${TOP_K:-3}"
AUDIO_MAX_LEN="${AUDIO_MAX_LEN:-128}"
VCODEC="${VCODEC:-h264}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

TRAIN_STEPS="${TRAIN_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
AUDIO_LR="${AUDIO_LR:-1e-4}"
LM_LR="${LM_LR:-}"
UNFREEZE_LM_LAYERS="${UNFREEZE_LM_LAYERS:-0}"
AUDIO_FUSION_MODE="${AUDIO_FUSION_MODE:-natural_language}"
CLASS_TOKEN_SCALE="${CLASS_TOKEN_SCALE:-0.1}"

SHUFFLE_SLOTS="${SHUFFLE_SLOTS:-on}"
# Leave target class flip noise enabled for all slots. This makes the mixed
# policy train against noisier class labels instead of assuming the target slot
# is always cleaner than distractors.
TARGET_SLOT_PROTECT="${TARGET_SLOT_PROTECT:-off}"
TARGET_FIRST_SLOTS="${TARGET_FIRST_SLOTS:-off}"
CANONICALIZE_SLOTS="${CANONICALIZE_SLOTS:-azimuth}"

NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"

IMAGE_COLOR_JITTER="${IMAGE_COLOR_JITTER:-0.05}"
IMAGE_TRANSLATE_PX="${IMAGE_TRANSLATE_PX:-2}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.003}"
AUDIO_CONF_DROPOUT="${AUDIO_CONF_DROPOUT:-0.0}"
DIRECTION_DROPOUT="${DIRECTION_DROPOUT:-0.05}"
DIRECTION_ENCODER_TYPE="${DIRECTION_ENCODER_TYPE:-mlp}"

VLM_LORA="${VLM_LORA:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"
RESUME_FROM="${RESUME_FROM:-}"

_warmup_default=$(( TRAIN_STEPS / 20 ))
if   (( _warmup_default > 500 )); then _warmup_default=500
elif (( _warmup_default < 1   )); then _warmup_default=1
fi
WARMUP_STEPS="${WARMUP_STEPS:-${_warmup_default}}"

_save_default=$(( TRAIN_STEPS / 10 ))
if (( _save_default < 1 )); then _save_default="${TRAIN_STEPS}"; fi
SAVE_EVERY="${SAVE_EVERY:-${_save_default}}"
LOG_EVERY="${LOG_EVERY:-200}"

EVAL_TASKS="${EVAL_TASKS:-both}"  # both | two | silent
EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"
EVAL_INTENTION_MODE="${EVAL_INTENTION_MODE:-exclusive}"
EVAL_EXCLUSIVE_INTENTION_MARGIN="${EVAL_EXCLUSIVE_INTENTION_MARGIN:-0.03}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-0}"
EVAL_NO_INSTRUCTION="${EVAL_NO_INSTRUCTION:-0}"
CLASS_SMOOTHING_WINDOW="${CLASS_SMOOTHING_WINDOW:-5}"
SHUFFLE_AUDIO_SLOTS_EVAL="${SHUFFLE_AUDIO_SLOTS_EVAL:-on}"
TARGET_FIRST_AUDIO_SLOTS_EVAL="${TARGET_FIRST_AUDIO_SLOTS_EVAL:-off}"
CANONICALIZE_AUDIO_SLOTS_EVAL="${CANONICALIZE_AUDIO_SLOTS_EVAL:-azimuth}"

PY="conda run -n ${CONDA_ENV} --no-capture-output python"
cd "${REPO_ROOT}"

cat <<EOF
[paths]    two_src   = ${TWO_SRC_DIR}
           silent    = ${SILENT_SRC_DIR}
           combined  = ${COMBINED_SRC_DIR}
           lerobot   = ${LEROBOT_DIR}
           output    = ${OUTPUT_DIR}
[stages]   combine=${DO_COMBINE} convert=${DO_CONVERT} train=${DO_TRAIN} eval=${DO_EVAL}
[stage 3]  steps=${TRAIN_STEPS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}
           lr=${LR} audio_lr=${AUDIO_LR} lm_lr=${LM_LR:-auto}
           unfreeze_lm=${UNFREEZE_LM_LAYERS} warmup=${WARMUP_STEPS}
           shuffle_slots=${SHUFFLE_SLOTS} target_slot_protect=${TARGET_SLOT_PROTECT}
           canonicalize_slots=${CANONICALIZE_SLOTS} direction_encoder=${DIRECTION_ENCODER_TYPE}
[stage 4]  eval_tasks=${EVAL_TASKS} n=${EVAL_N} save_video=${EVAL_SAVE_VIDEO}
EOF

if [[ "${DO_COMBINE}" == "1" ]]; then
    echo "============================================================"
    echo "[1/4] combining HDF5 datasets"
    echo "============================================================"
    rm -rf "${COMBINED_SRC_DIR}"
    mkdir -p "${COMBINED_SRC_DIR}"
    ${PY} - <<PY
from pathlib import Path
import os
import shutil

sources = [
    ("two", Path("${TWO_SRC_DIR}")),
    ("silent", Path("${SILENT_SRC_DIR}")),
]
out = Path("${COMBINED_SRC_DIR}")
idx = 0
counts = {}
for label, src in sources:
    files = sorted(src.glob("data_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no data_*.hdf5 in {src}")
    counts[label] = len(files)
    for path in files:
        dst = out / f"data_{idx:06d}.hdf5"
        try:
            os.link(path, dst)
        except OSError:
            shutil.copy2(path, dst)
        idx += 1
print({"counts": counts, "total": idx, "out": str(out)})
PY
fi

if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "============================================================"
    echo "[2/4] converting mixed HDF5 → ${LEROBOT_DIR}"
    echo "============================================================"
    rm -rf "${LEROBOT_DIR}"
    ${PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${COMBINED_SRC_DIR}" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps 10 \
        --top-k "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec "${VCODEC}" \
        --oracle-mode \
        --use-episode-instruction
fi

if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "============================================================"
    echo "[3/4] training mixed policy (${TRAIN_STEPS} steps)"
    echo "============================================================"
    if [[ -n "${RESUME_FROM}" ]]; then
        mkdir -p "${OUTPUT_DIR}"
    else
        rm -rf "${OUTPUT_DIR}"
    fi
    lm_lr_args=()
    [[ -n "${LM_LR}" ]] && lm_lr_args+=(--lm-lr "${LM_LR}")
    lora_args=()
    if [[ "${VLM_LORA}" == "1" ]]; then
        lora_args+=(
            --vlm-lora
            --lora-r "${LORA_R}"
            --lora-alpha "${LORA_ALPHA}"
            --lora-dropout "${LORA_DROPOUT}"
            --lora-target-modules "${LORA_TARGET_MODULES}"
        )
    fi
    resume_args=()
    [[ -n "${RESUME_FROM}" ]] && resume_args+=(--resume-from "${RESUME_FROM}")
    ${PY} src/training/train_smolvla_audio.py \
        --dataset-root "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --pretrained "${PRETRAINED}" \
        --taxonomy "${TAXONOMY}" \
        --output-dir "${OUTPUT_DIR}" \
        --audio-max-len "${AUDIO_MAX_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        --steps "${TRAIN_STEPS}" \
        --lr "${LR}" \
        --audio-lr "${AUDIO_LR}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --num-workers "${NUM_WORKERS}" \
        --save-every "${SAVE_EVERY}" \
        --log-every "${LOG_EVERY}" \
        --unfreeze-last-n-lm-layers "${UNFREEZE_LM_LAYERS}" \
        --audio-fusion-mode "${AUDIO_FUSION_MODE}" \
        --class-token-scale "${CLASS_TOKEN_SCALE}" \
        --oracle-noise on \
        --noise-az-std "${NOISE_AZ_STD}" \
        --noise-el-std "${NOISE_EL_STD}" \
        --noise-conf-min "${NOISE_CONF_MIN}" \
        --noise-conf-max "${NOISE_CONF_MAX}" \
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
        --shuffle-slots "${SHUFFLE_SLOTS}" \
        --target-slot-protect "${TARGET_SLOT_PROTECT}" \
        --target-first-slots "${TARGET_FIRST_SLOTS}" \
        --canonicalize-slots "${CANONICALIZE_SLOTS}" \
        --image-color-jitter "${IMAGE_COLOR_JITTER}" \
        --image-translate-px "${IMAGE_TRANSLATE_PX}" \
        --state-noise-std "${STATE_NOISE_STD}" \
        --audio-conf-dropout "${AUDIO_CONF_DROPOUT}" \
        --direction-dropout "${DIRECTION_DROPOUT}" \
        --direction-encoder-type "${DIRECTION_ENCODER_TYPE}" \
        "${lm_lr_args[@]}" \
        "${lora_args[@]}" \
        "${resume_args[@]}"
fi

run_oracle_eval() {
    local task_name="$1"
    local save_dir="$2"
    local ckpt="$3"
    extra=()
    [[ "${EVAL_SAVE_VIDEO}" == "1" ]] && extra+=(--save-video)
    [[ "${EVAL_NO_INSTRUCTION}" == "1" ]] && extra+=(--no-instruction)
    echo "============================================================"
    echo "[4/4] eval ORACLE ${task_name} (${EVAL_N} episodes)"
    echo "       ckpt=${ckpt}"
    echo "       out =${save_dir}"
    echo "============================================================"
    ${PY} src/eval/eval_smolvla_audio.py \
        --ckpt "${ckpt}" \
        --pretrained "${PRETRAINED}" \
        --taxonomy "${TAXONOMY}" \
        --task-name "${task_name}" \
        --n-episodes "${EVAL_N}" \
        --max-episode-length "${EVAL_MAX_LEN}" \
        --horizon "${EVAL_HORIZON}" \
        --intention-threshold "${EVAL_INTENTION_THRESHOLD}" \
        --intention-mode "${EVAL_INTENTION_MODE}" \
        --exclusive-intention-margin "${EVAL_EXCLUSIVE_INTENTION_MARGIN}" \
        --save-dir "${save_dir}" \
        --oracle-mode \
        --noise-az-std "${NOISE_AZ_STD}" \
        --noise-el-std "${NOISE_EL_STD}" \
        --noise-conf-min "${NOISE_CONF_MIN}" \
        --noise-conf-max "${NOISE_CONF_MAX}" \
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
        --class-smoothing-window "${CLASS_SMOOTHING_WINDOW}" \
        --shuffle-audio-slots "${SHUFFLE_AUDIO_SLOTS_EVAL}" \
        --target-first-audio-slots "${TARGET_FIRST_AUDIO_SLOTS_EVAL}" \
        --canonicalize-audio-slots "${CANONICALIZE_AUDIO_SLOTS_EVAL}" \
        "${extra[@]}"
}

if [[ "${DO_EVAL}" == "1" ]]; then
    CKPT="$(ls -t ${OUTPUT_DIR}/ckpt_step*.pt 2>/dev/null | head -1)"
    if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
        echo "[err] no ckpt found in ${OUTPUT_DIR}"
        exit 1
    fi
    case "${EVAL_TASKS}" in
        both)
            run_oracle_eval "select_radio_two" "${EVAL_DIR_TWO}" "${CKPT}"
            run_oracle_eval "select_radio_silent" "${EVAL_DIR_SILENT}" "${CKPT}"
            ;;
        two)
            run_oracle_eval "select_radio_two" "${EVAL_DIR_TWO}" "${CKPT}"
            ;;
        silent)
            run_oracle_eval "select_radio_silent" "${EVAL_DIR_SILENT}" "${CKPT}"
            ;;
        *)
            echo "[err] EVAL_TASKS must be one of: both | two | silent"
            exit 1
            ;;
    esac
fi

echo "============================================================"
echo "[done] mixed two-radio pipeline finished"
echo "  combined HDF5 : ${COMBINED_SRC_DIR}"
echo "  LeRobot       : ${LEROBOT_DIR}"
echo "  checkpoints   : ${OUTPUT_DIR}"
echo "  eval two      : ${EVAL_DIR_TWO}"
echo "  eval silent   : ${EVAL_DIR_SILENT}"
echo "============================================================"

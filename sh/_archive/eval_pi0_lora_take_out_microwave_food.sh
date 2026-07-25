#!/usr/bin/env bash
# Evaluate the LoRA-fine-tuned Pi0 take_out_microwave_food policy.
#
# Pairs with sh/train_pi0_lora_take_out_microwave_food.sh. The trained config
# uses data.audio_mode="slots", so eval MUST pass --audio-mode slots (the
# generic sh/eval_pi05_audio.sh wrapper defaults to "text" and is not used
# here). LoRA adapters are auto-injected from the checkpoint's lora_config.json,
# so no LoRA flags are needed on the eval side.
#
# Usage (defaults eval the final 60000-step checkpoint, oracle SELD, n=30):
#   bash sh/eval_pi0_lora_take_out_microwave_food.sh
#
# Override the checkpoint step / episode count:
#   STEP=40000 EVAL_N=5 EVAL_SAVE_VIDEO=1 bash sh/eval_pi0_lora_take_out_microwave_food.sh
#
# Sweep several steps:
#   for s in 20000 40000 60000; do STEP=$s bash sh/eval_pi0_lora_take_out_microwave_food.sh; done
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_take_out_microwave_food_lora}"
EXP_NAME="${EXP_NAME:-pi0_lora_take_out_microwave_food}"
CKPT_BASE="${CKPT_BASE:-${REPO_ROOT}/outputs/pi0_lora_take_out_microwave_food}"
STEP="${STEP:-60000}"
POLICY_DIR="${POLICY_DIR:-${CKPT_BASE}/${POLICY_CONFIG}/${EXP_NAME}/${STEP}}"

TASK_NAME="${TASK_NAME:-take_out_microwave_food}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
AUDIO_MODE="${AUDIO_MODE:-slots}"

EVAL_MODE="${EVAL_MODE:-oracle}"   # oracle | real_sled
EVAL_N="${EVAL_N:-30}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-550}"
EVAL_HORIZON="${EVAL_HORIZON:-10}"
TOP_K="${TOP_K:-3}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi0_lora_take_out_microwave_food/step${STEP}_n${EVAL_N}}"

EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"
EVAL_INTENTION_MODE="${EVAL_INTENTION_MODE:-exclusive}"
EVAL_EXCLUSIVE_INTENTION_MARGIN="${EVAL_EXCLUSIVE_INTENTION_MARGIN:-0.03}"
CLASS_SMOOTHING_WINDOW="${CLASS_SMOOTHING_WINDOW:-5}"
SHUFFLE_AUDIO_SLOTS_EVAL="${SHUFFLE_AUDIO_SLOTS_EVAL:-on}"
TARGET_FIRST_AUDIO_SLOTS_EVAL="${TARGET_FIRST_AUDIO_SLOTS_EVAL:-off}"
CANONICALIZE_AUDIO_SLOTS_EVAL="${CANONICALIZE_AUDIO_SLOTS_EVAL:-azimuth}"

EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_NO_INSTRUCTION="${EVAL_NO_INSTRUCTION:-0}"

# Robot-motion-vs-sound diagnostic. MOTION_THRESHOLD is the primary
# "did it move?" bar (m of EE displacement from rest); MOTION_THRESHOLDS sweeps
# extra (relaxed→strict) bars in the same rollout (no re-run needed).
MOTION_THRESHOLD="${MOTION_THRESHOLD:-0.02}"
MOTION_THRESHOLDS="${MOTION_THRESHOLDS:-0.01,0.02,0.05,0.1,0.2}"
SAVE_MOTION_TRACE="${SAVE_MOTION_TRACE:-1}"

# Oracle SELD noise (matches the frozen-pipeline eval defaults).
NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"

# real_sled mode inputs.
AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"
EVAL_WARMUP_SECONDS="${EVAL_WARMUP_SECONDS:-3.0}"

# Match the training-time chime distribution.
export VLABENCH_MICROWAVE_MIN_DELAY_SEC="${VLABENCH_MICROWAVE_MIN_DELAY_SEC:-0.0}"
export VLABENCH_MICROWAVE_MAX_DELAY_SEC="${VLABENCH_MICROWAVE_MAX_DELAY_SEC:-20.0}"
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC="${VLABENCH_MICROWAVE_REACT_WINDOW_SEC:-15.0}"

cd "${REPO_ROOT}"

if [[ ! -d "${POLICY_DIR}" ]]; then
    echo "[err] checkpoint dir not found: ${POLICY_DIR}"
    echo "      set STEP=<one of $(ls "${CKPT_BASE}/${POLICY_CONFIG}/${EXP_NAME}" 2>/dev/null | tr '\n' ' ')>"
    exit 1
fi

extra=()
[[ "${EVAL_SAVE_VIDEO}" == "1" ]] && extra+=(--save-video)
[[ "${EVAL_NO_INSTRUCTION}" == "1" ]] && extra+=(--no-instruction)
[[ "${SAVE_MOTION_TRACE}" == "1" ]] && extra+=(--save-motion-trace)

mode_args=()
if [[ "${EVAL_MODE}" == "oracle" ]]; then
    mode_args+=(
        --oracle-mode
        --noise-az-std "${NOISE_AZ_STD}"
        --noise-el-std "${NOISE_EL_STD}"
        --noise-conf-min "${NOISE_CONF_MIN}"
        --noise-conf-max "${NOISE_CONF_MAX}"
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}"
    )
elif [[ "${EVAL_MODE}" == "real_sled" ]]; then
    mode_args+=(
        --audio-config "${AUDIO_CONFIG}"
        --sled-ckpt "${SLED_CKPT}"
        --warmup-seconds "${EVAL_WARMUP_SECONDS}"
    )
else
    echo "[err] EVAL_MODE must be oracle or real_sled"
    exit 1
fi

echo "============================================================"
echo "[eval] pi0 LoRA take_out_microwave_food"
echo "       config=${POLICY_CONFIG}  audio_mode=${AUDIO_MODE}"
echo "       policy_dir=${POLICY_DIR}"
echo "       task=${TASK_NAME} mode=${EVAL_MODE} n=${EVAL_N}"
echo "       out=${EVAL_DIR}"
echo "============================================================"

env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    uv --project "${OPENPI_ROOT}" run python \
    "${REPO_ROOT}/src/eval/eval_pi05_audio.py" \
    --local \
    --policy-config "${POLICY_CONFIG}" \
    --policy-dir "${POLICY_DIR}" \
    --openpi-root "${OPENPI_ROOT}" \
    --taxonomy "${TAXONOMY}" \
    --task-name "${TASK_NAME}" \
    --audio-mode "${AUDIO_MODE}" \
    --n-episodes "${EVAL_N}" \
    --max-episode-length "${EVAL_MAX_LEN}" \
    --horizon "${EVAL_HORIZON}" \
    --top-k "${TOP_K}" \
    --intention-threshold "${EVAL_INTENTION_THRESHOLD}" \
    --intention-mode "${EVAL_INTENTION_MODE}" \
    --exclusive-intention-margin "${EVAL_EXCLUSIVE_INTENTION_MARGIN}" \
    --save-dir "${EVAL_DIR}" \
    --class-smoothing-window "${CLASS_SMOOTHING_WINDOW}" \
    --shuffle-audio-slots "${SHUFFLE_AUDIO_SLOTS_EVAL}" \
    --target-first-audio-slots "${TARGET_FIRST_AUDIO_SLOTS_EVAL}" \
    --canonicalize-audio-slots "${CANONICALIZE_AUDIO_SLOTS_EVAL}" \
    --motion-threshold "${MOTION_THRESHOLD}" \
    --motion-thresholds "${MOTION_THRESHOLDS}" \
    "${mode_args[@]}" \
    "${extra[@]}"

echo "[done] results -> ${EVAL_DIR}/summary.json"

#!/usr/bin/env bash
# Evaluate an audio-conditioned pi0.5/openpi policy on AVLABench radio tasks.
#
# Local mode (default) loads openpi/pi0.5 in this process:
#   LOCAL=1 POLICY_CONFIG=pi05_vlabench_primitive_lora \
#   POLICY_DIR=/path/to/pi05/checkpoint bash sh/eval_pi05_audio.sh
#
# Server mode is still available:
#   cd third_party/openpi
#   uv run scripts/serve_policy.py --env VLABENCH \
#     policy:checkpoint \
#     --policy.config=pi05_vlabench_primitive_lora \
#     --policy.dir=/path/to/pi05/checkpoint
#   LOCAL=0 HOST=localhost PORT=8000 bash sh/eval_pi05_audio.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

LOCAL="${LOCAL:-1}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_ft_vlabench_primitive}"
POLICY_DIR="${POLICY_DIR:-}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
TASK_NAME="${TASK_NAME:-select_radio_two}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"

EVAL_MODE="${EVAL_MODE:-oracle}"  # oracle | real_sled
AUDIO_MODE="${AUDIO_MODE:-text}"  # text | slots | slots_uv (match training config)
EVAL_N="${EVAL_N:-20}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi05_audio}"
TOP_K="${TOP_K:-3}"

EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"
EVAL_INTENTION_MODE="${EVAL_INTENTION_MODE:-exclusive}"
EVAL_EXCLUSIVE_INTENTION_MARGIN="${EVAL_EXCLUSIVE_INTENTION_MARGIN:-0.03}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-0}"
EVAL_NO_INSTRUCTION="${EVAL_NO_INSTRUCTION:-0}"

NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"
NOISE_ENERGY_STD="${NOISE_ENERGY_STD:-0.05}"

CLASS_SMOOTHING_WINDOW="${CLASS_SMOOTHING_WINDOW:-5}"
SHUFFLE_AUDIO_SLOTS_EVAL="${SHUFFLE_AUDIO_SLOTS_EVAL:-on}"
TARGET_FIRST_AUDIO_SLOTS_EVAL="${TARGET_FIRST_AUDIO_SLOTS_EVAL:-off}"
CANONICALIZE_AUDIO_SLOTS_EVAL="${CANONICALIZE_AUDIO_SLOTS_EVAL:-azimuth}"

AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"
EVAL_WARMUP_SECONDS="${EVAL_WARMUP_SECONDS:-3.0}"
EVAL_SAVE_AUDIO="${EVAL_SAVE_AUDIO:-0}"
EVAL_SAVE_AUDIO_LOG="${EVAL_SAVE_AUDIO_LOG:-0}"

cd "${REPO_ROOT}"

extra=()
[[ "${EVAL_SAVE_VIDEO}" == "1" ]] && extra+=(--save-video)
[[ "${EVAL_SAVE_AUDIO}" == "1" ]] && extra+=(--save-audio)
[[ "${EVAL_SAVE_AUDIO_LOG}" == "1" ]] && extra+=(--save-audio-log)
[[ "${EVAL_NO_INSTRUCTION}" == "1" ]] && extra+=(--no-instruction)

mode_args=()
if [[ "${EVAL_MODE}" == "oracle" ]]; then
    mode_args+=(
        --oracle-mode
        --noise-az-std "${NOISE_AZ_STD}"
        --noise-el-std "${NOISE_EL_STD}"
        --noise-conf-min "${NOISE_CONF_MIN}"
        --noise-conf-max "${NOISE_CONF_MAX}"
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}"
        --noise-energy-std "${NOISE_ENERGY_STD}"
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
echo "[eval] pi0.5/openpi audio policy"
if [[ "${LOCAL}" == "1" ]]; then
    echo "       mode=local config=${POLICY_CONFIG}"
    echo "       policy_dir=${POLICY_DIR}"
    echo "       openpi_root=${OPENPI_ROOT}"
else
    echo "       mode=server server=${HOST}:${PORT}"
fi
echo "       task=${TASK_NAME} mode=${EVAL_MODE} n=${EVAL_N}"
echo "       out=${EVAL_DIR}"
echo "============================================================"

local_args=()
if [[ "${LOCAL}" == "1" ]]; then
    if [[ -z "${POLICY_DIR}" ]]; then
        echo "[err] LOCAL=1 requires POLICY_DIR=/path/to/openpi/checkpoint"
        exit 1
    fi
    local_args+=(
        --local
        --policy-config "${POLICY_CONFIG}"
        --policy-dir "${POLICY_DIR}"
        --openpi-root "${OPENPI_ROOT}"
    )
else
    local_args+=(--server)
fi

if [[ "${LOCAL}" == "1" ]]; then
    RUN_PY=(env "UV_CACHE_DIR=${UV_CACHE_DIR}" uv --project "${OPENPI_ROOT}" run python)
else
    RUN_PY=(conda run -n "${CONDA_ENV}" --no-capture-output python)
fi

"${RUN_PY[@]}" "${REPO_ROOT}/src/eval/eval_pi05_audio.py" \
    --host "${HOST}" \
    --port "${PORT}" \
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
    "${local_args[@]}" \
    "${mode_args[@]}" \
    "${extra[@]}"

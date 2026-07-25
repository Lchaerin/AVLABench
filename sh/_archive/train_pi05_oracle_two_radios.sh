#!/usr/bin/env bash
# pi0.5/openpi variant of the two-radio oracle pipeline.
#
# Data generation/conversion stay identical to the SmolVLA pipeline. Training
# is delegated to openpi because the pi0.5 backbone lives in third_party/openpi.
# This script prepares data and runs evaluation with audio information injected
# as prompt text. Evaluation defaults to local openpi loading, no server needed.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_oracle_two}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_oracle_two_lerobot}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi05_oracle_two}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"

CONDA_ENV="${CONDA_ENV:-vlabench}"
PY="${PY:-conda run -n ${CONDA_ENV} --no-capture-output python}"
REPO_ID="${REPO_ID:-local/avla_select_radio_two_oracle}"

DO_GENERATE="${DO_GENERATE:-0}"
DO_CONVERT="${DO_CONVERT:-0}"
DO_TRAIN="${DO_TRAIN:-0}"
DO_EVAL="${DO_EVAL:-1}"

N_SAMPLE="${N_SAMPLE:-500}"
TASK_NAME="${TASK_NAME:-select_radio_two}"
MAX_EPISODE="${MAX_EPISODE:-${N_SAMPLE}}"
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"

TOP_K="${TOP_K:-3}"
VCODEC="${VCODEC:-h264}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

LOCAL="${LOCAL:-1}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_vlabench_primitive_lora}"
POLICY_DIR="${POLICY_DIR:-}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
EVAL_MODE="${EVAL_MODE:-oracle}"
EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"

cd "${REPO_ROOT}"

cat <<EOF
[paths]  repo=${REPO_ROOT}
         dataset=${DATASET_DIR}
         lerobot=${LEROBOT_DIR}
         eval=${EVAL_DIR}
[stages] gen=${DO_GENERATE} convert=${DO_CONVERT} train=${DO_TRAIN} eval=${DO_EVAL}
[pi0.5]  local=${LOCAL} config=${POLICY_CONFIG}
         policy_dir=${POLICY_DIR}
         openpi_root=${OPENPI_ROOT}
EOF

if [[ "${DO_GENERATE}" == "1" ]]; then
    ${PY} scripts/trajectory_generation.py \
        --task-name "${TASK_NAME}" \
        --oracle-mode \
        --save-dir "${DATASET_DIR}" \
        --n-sample "${N_SAMPLE}" \
        --max-episode "${MAX_EPISODE}" \
        --start-idle-seconds "${START_IDLE_SECONDS}" \
        --dataset-fps "${DATASET_FPS}"
fi

if [[ "${DO_CONVERT}" == "1" ]]; then
    rm -rf "${LEROBOT_DIR}"
    ${PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${DATASET_DIR}/${TASK_NAME}" \
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
    cat <<EOF
[train] pi0.5 fine-tuning is handled by openpi, not by src/training/train_smolvla_audio.py.
        Use ${LEROBOT_DIR} / repo-id ${REPO_ID} as the LeRobot dataset for the
        openpi pi0.5 training config. For evaluation, set POLICY_DIR to the
        resulting checkpoint and keep LOCAL=1, or set LOCAL=0 to use a server.
EOF
fi

if [[ "${DO_EVAL}" == "1" ]]; then
    EVAL_DIR="${EVAL_DIR}" \
    TASK_NAME="${TASK_NAME}" \
    TAXONOMY="${TAXONOMY}" \
    LOCAL="${LOCAL}" \
    POLICY_CONFIG="${POLICY_CONFIG}" \
    POLICY_DIR="${POLICY_DIR}" \
    OPENPI_ROOT="${OPENPI_ROOT}" \
    HOST="${HOST}" \
    PORT="${PORT}" \
    EVAL_MODE="${EVAL_MODE}" \
    EVAL_N="${EVAL_N}" \
    EVAL_MAX_LEN="${EVAL_MAX_LEN}" \
    EVAL_HORIZON="${EVAL_HORIZON}" \
    EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO}" \
    TOP_K="${TOP_K}" \
    bash sh/eval_pi05_audio.sh
fi

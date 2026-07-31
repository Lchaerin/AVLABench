#!/usr/bin/env bash
# Rebuild the find_hidden dataset with the corrected camera layout, then train.
#
# Why: camera 2 ("forward") used to sit BEHIND the cabinets, so the primary
# image the policy conditions on showed only their back panels -- the handles it
# has to grasp were in no camera. Camera 2 is also the microphone / uv reference
# (see src/audio/oracle_sled.py), so it was moved to an over-the-shoulder pose on
# the robot's side rather than top-down, which would have collapsed the
# elevation cue that distinguishes the top and bottom drawer. Camera 0
# ("second_image") became the top-down layout view. Both live in
# VLABench/configs/camera_config.json, so the images AND the oracle audio GT are
# regenerated consistently -- which is why the dataset must be rebuilt.
#
# Stages: wait for a GPU-hogging PID (optional) -> generate -> convert -> norm
#         -> train -> eval. Set WAIT_PID=0 to start immediately.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
cd "${REPO_ROOT}"

WAIT_PID="${WAIT_PID:-0}"
PER_SLOT="${PER_SLOT:-220}"          # 4 slots x 220 = 880, matching the v1 set
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_find_hidden_v2_src}"
TASK="find_hidden_object_open"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_find_hidden_lerobot}"

BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EXP_NAME="${EXP_NAME:-pi0_find_hidden_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_find_hidden_v2}"

export OPENPI_PI0_JAX_WEIGHT="${OPENPI_PI0_JAX_WEIGHT:-${REPO_ROOT}/checkpoints/pi0_base_primitive/params}"
export OPENPI_PI0_PYTORCH_WEIGHT="${OPENPI_PI0_PYTORCH_WEIGHT:-${REPO_ROOT}/checkpoints/pi0_base_primitive_torch}"

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; exit 1; }

# ---------------------------------------------------------------------------
# [0] wait for the GPU to free up (the in-flight eval), if asked
# ---------------------------------------------------------------------------
if [[ "${WAIT_PID}" != "0" ]]; then
    say "waiting for PID ${WAIT_PID} (in-flight eval) to exit..."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do sleep 60; done
    say "PID ${WAIT_PID} gone; continuing"
    sleep 20   # let its GPU memory actually be released
fi

# ---------------------------------------------------------------------------
# [1] generate with the new cameras
# ---------------------------------------------------------------------------
say "=== [1] generate ${PER_SLOT}/slot into ${GEN_ROOT} (MAX_PARALLEL=${MAX_PARALLEL})"
GEN_ROOT="${GEN_ROOT}" \
STAGING="${REPO_ROOT}/dataset_find_hidden_v2_staging" \
LOGDIR="${REPO_ROOT}/outputs/gen_find_hidden_v2_logs" \
PER_SLOT="${PER_SLOT}" MAX_PARALLEL="${MAX_PARALLEL}" \
    bash sh/gen_find_hidden_balance.sh || say "WARN: generator returned non-zero (partial slots are still usable)"

n_ep=$(ls "${GEN_ROOT}/${TASK}"/*.hdf5 2>/dev/null | wc -l)
say "generated episodes: ${n_ep}"
(( n_ep >= 400 )) || die "only ${n_ep} episodes generated (expected ~$((PER_SLOT*4))); refusing to train"

# ---------------------------------------------------------------------------
# [2-5] convert -> norm stats -> train -> eval
# ---------------------------------------------------------------------------
# Keep the v1 LeRobot dataset around; the converter would wipe it in place.
if [[ -d "${LEROBOT_DIR}" && ! -e "${LEROBOT_DIR}_v1" ]]; then
    mv "${LEROBOT_DIR}" "${LEROBOT_DIR}_v1" && say "kept previous LeRobot set as ${LEROBOT_DIR}_v1"
fi

say "=== [2-5] convert -> norm -> train (batch=${BATCH_SIZE}, steps=${TRAIN_STEPS}) -> eval"
DO_GENERATE=0 DO_CONVERT=1 DO_NORM=1 DO_TRAIN=1 DO_EVAL=1 \
GEN_ROOT="${GEN_ROOT}" SRC_DIR="${GEN_ROOT}/${TASK}" \
LEROBOT_DIR="${LEROBOT_DIR}" \
VLM_LORA=1 \
BATCH_SIZE="${BATCH_SIZE}" TRAIN_STEPS="${TRAIN_STEPS}" \
DECAY_STEPS="${TRAIN_STEPS}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
NUM_WORKERS=8 \
EXP_NAME="${EXP_NAME}" OUTPUT_DIR="${OUTPUT_DIR}" \
EVAL_SAVE_VIDEO=1 \
    bash sh/train_pi0_find_hidden.sh || die "train pipeline returned non-zero"

say "=== DONE: checkpoints under ${OUTPUT_DIR}"

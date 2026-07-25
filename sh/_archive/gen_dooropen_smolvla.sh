#!/usr/bin/env bash
# TEMPORARY simplified take_out_microwave_food pipeline: episode ends when the
# microwave door is opened after the chime (VLABENCH_MICROWAVE_DOOROPEN_ONLY=1).
# Generates oracle trajectories where the expert WAITS for the chime then opens
# the door — the fix for the "policy ignores audio timing" finding.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
PY="conda run -n ${CONDA_ENV} --no-capture-output python"

DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_dooropen}"
N_SAMPLE="${N_SAMPLE:-220}"
MAX_EPISODE="${MAX_EPISODE:-130}"
START_IDLE="${START_IDLE:-1.0}"

export MUJOCO_GL=egl
export VLABENCH_MICROWAVE_DOOROPEN_ONLY=1
export VLABENCH_MICROWAVE_PREGRIP=0
export VLABENCH_MICROWAVE_MIN_DELAY_SEC="${VLABENCH_MICROWAVE_MIN_DELAY_SEC:-2.0}"
export VLABENCH_MICROWAVE_MAX_DELAY_SEC="${VLABENCH_MICROWAVE_MAX_DELAY_SEC:-6.0}"
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC="${VLABENCH_MICROWAVE_REACT_WINDOW_SEC:-15.0}"

cd "${REPO_ROOT}"
echo "[gen] dir=${DATASET_DIR} n_sample=${N_SAMPLE} max_episode=${MAX_EPISODE}"
${PY} scripts/trajectory_generation.py \
    --task-name take_out_microwave_food \
    --save-dir "${DATASET_DIR}" \
    --n-sample "${N_SAMPLE}" \
    --max-episode "${MAX_EPISODE}" \
    --oracle-mode \
    --start-idle-seconds "${START_IDLE}"

n=$(ls "${DATASET_DIR}/take_out_microwave_food"/data_*.hdf5 2>/dev/null | wc -l)
echo "[gen] DONE: ${n} successful episodes in ${DATASET_DIR}/take_out_microwave_food"

#!/usr/bin/env bash
# End-to-end Pi0 (openpi backbone) pipeline for take_out_microwave_food.
#
# Why Pi0 instead of SmolVLA?
# ---------------------------
#   SmolVLA is small (~450 M) and struggles with the precise long-horizon
#   manipulation this task requires (gripper visibly jitters near the
#   handle, never completes the open_door → pick → place chain). Pi0
#   shares the same VLA pattern but has a larger Gemma 2B backbone +
#   bigger action expert, which gives the policy enough capacity to keep
#   the trajectory tight across the ~380-frame demos.
#
# Pipeline layout
# ---------------
#   Stage 1 (trajectory generation)  identical to the SmolVLA pipeline
#   Stage 2 (LeRobot conversion)     identical
#   Stage 3 (training)               openpi/scripts/train_pytorch.py
#   Stage 4 (eval)                   delegated to sh/eval_pi05_audio.sh
#                                    (same Python entrypoint handles pi0
#                                    + pi05; my eval-script fixes for
#                                    take_out_microwave_food apply here)
#
# REQUIRED env vars (Pi0 base weights)
# ------------------------------------
#   OPENPI_PI0_JAX_WEIGHT       JAX/Flax pi0_base params dir (used during
#                               weight loading on Stage 3)
#   OPENPI_PI0_PYTORCH_WEIGHT   PyTorch port of pi0_base for the
#                               train_pytorch.py loader
#
# Both are read by the `pi0_ft_vlabench_take_out_microwave_food` config
# in `third_party/openpi/src/openpi/training/config.py`. Download them
# from openpi-assets if you haven't already:
#   https://github.com/Physical-Intelligence/openpi#downloading-weights
#
# Stage toggles:  DO_GENERATE / DO_CONVERT / DO_TRAIN / DO_EVAL  (0/1)

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"

# Reuse the oracle-mode dataset directories produced by the SmolVLA
# pipeline — they hold the same HDF5 + LeRobot data that Pi0 needs.
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_take_out_microwave_food_oracle}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_take_out_microwave_food_oracle_lerobot}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_take_out_microwave_food}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi0_take_out_microwave_food}"

TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"

CONDA_ENV="${CONDA_ENV:-vlabench}"
PY="${PY:-conda run -n ${CONDA_ENV} --no-capture-output python}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

# Stage 2 (HDF5 → LeRobot) MUST use openpi's pinned lerobot, not the conda
# vlabench env. The vlabench env ships lerobot 0.4.x which writes the v3.0
# on-disk layout (meta/tasks.parquet, meta/episodes/*.parquet), but openpi's
# .venv pins lerobot 0.1.0 which only reads the v2.x layout
# (meta/tasks.jsonl, meta/episodes.jsonl). Converting with the wrong lerobot
# produces a dataset train_pytorch.py cannot load (it then falls back to the
# HF hub and 404s). `uv run` inside OPENPI_ROOT resolves to that .venv, and
# the convert script's other deps (cv2, h5py, src.audio, VLABench.utils) are
# all importable there.
CONVERT_PY="${CONVERT_PY:-env UV_CACHE_DIR=${UV_CACHE_DIR} uv --project ${OPENPI_ROOT} run python}"

REPO_ID="${REPO_ID:-local/avla_take_out_microwave_food}"
TASK_NAME="${TASK_NAME:-take_out_microwave_food}"

# ---------------------------------------------------------------------------
# Stage toggles
# ---------------------------------------------------------------------------
DO_GENERATE="${DO_GENERATE:-0}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

# ---------------------------------------------------------------------------
# Task-specific timing knobs (same shape as the SmolVLA pipeline so the
# train- and eval-time chime distributions stay aligned).
# ---------------------------------------------------------------------------
MIN_DELAY_SEC="${MIN_DELAY_SEC:-0.0}"
MAX_DELAY_SEC="${MAX_DELAY_SEC:-20.0}"
REACT_WINDOW_SEC="${REACT_WINDOW_SEC:-15.0}"
export VLABENCH_MICROWAVE_MIN_DELAY_SEC="${MIN_DELAY_SEC}"
export VLABENCH_MICROWAVE_MAX_DELAY_SEC="${MAX_DELAY_SEC}"
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC="${REACT_WINDOW_SEC}"

# ---------------------------------------------------------------------------
# Stage 1 knobs
# ---------------------------------------------------------------------------
N_SAMPLE="${N_SAMPLE:-2000}"
MAX_EPISODE="${MAX_EPISODE:-${N_SAMPLE}}"
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"

# ---------------------------------------------------------------------------
# Stage 2 knobs
# ---------------------------------------------------------------------------
TOP_K="${TOP_K:-3}"
VCODEC="${VCODEC:-h264}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# ---------------------------------------------------------------------------
# Stage 3 knobs — openpi training
# ---------------------------------------------------------------------------
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_take_out_microwave_food}"
EXP_NAME="${EXP_NAME:-pi0_take_out_microwave_food}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUME="${RESUME:-0}"

# Partial fine-tuning toggle: freeze the vision tower + LLM half of Pi0 and
# only train the action expert + projection heads. Required to keep AdamW
# state small enough to fit Pi0 (~2.4B params) on a single 32GB GPU.
FREEZE_PALIGEMMA="${FREEZE_PALIGEMMA:-1}"

# ---------------------------------------------------------------------------
# Stage 4 knobs (delegated to eval_pi05_audio.sh, which uses POLICY_DIR)
# ---------------------------------------------------------------------------
EVAL_MODE="${EVAL_MODE:-oracle}"     # oracle | real_sled
EVAL_N="${EVAL_N:-100}"
_eval_max_default=$(python3 -c "import math; print(int(math.ceil((${MAX_DELAY_SEC} + ${REACT_WINDOW_SEC}) * 10) + 200))")
EVAL_MAX_LEN="${EVAL_MAX_LEN:-${_eval_max_default}}"
EVAL_HORIZON="${EVAL_HORIZON:-10}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"

POLICY_DIR="${POLICY_DIR:-${OUTPUT_DIR}/${EXP_NAME}/checkpoints/${TRAIN_STEPS}}"

cd "${REPO_ROOT}"

cat <<EOF
[paths]   repo      = ${REPO_ROOT}
          openpi    = ${OPENPI_ROOT}
          dataset   = ${DATASET_DIR}
          lerobot   = ${LEROBOT_DIR}
          output    = ${OUTPUT_DIR}
          eval_dir  = ${EVAL_DIR}
[task]    ${TASK_NAME}
          chime delay   ~ U[${MIN_DELAY_SEC}, ${MAX_DELAY_SEC}] s
          react window  = ${REACT_WINDOW_SEC} s
[stages]  gen=${DO_GENERATE} convert=${DO_CONVERT} train=${DO_TRAIN} eval=${DO_EVAL}
[stage 1] n_sample=${N_SAMPLE} cap=${MAX_EPISODE}
[stage 3] policy_config=${POLICY_CONFIG}  exp_name=${EXP_NAME}
          steps=${TRAIN_STEPS}  batch=${BATCH_SIZE}  gpus=${NUM_GPUS}
          OPENPI_PI0_JAX_WEIGHT     = ${OPENPI_PI0_JAX_WEIGHT:-(unset)}
          OPENPI_PI0_PYTORCH_WEIGHT = ${OPENPI_PI0_PYTORCH_WEIGHT:-(unset)}
[stage 4] mode=${EVAL_MODE}  n=${EVAL_N}  max_len=${EVAL_MAX_LEN}  horizon=${EVAL_HORIZON}
          policy_dir=${POLICY_DIR}
EOF

# ---------------------------------------------------------------------------
# Stage 1 — trajectory generation (oracle mode, same as SmolVLA pipeline)
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    echo "============================================================"
    echo "[1/4] generating up to ${N_SAMPLE} ORACLE trajectories (cap=${MAX_EPISODE})"
    echo "============================================================"
    ${PY} scripts/trajectory_generation.py \
        --task-name    "${TASK_NAME}" \
        --oracle-mode \
        --save-dir     "${DATASET_DIR}" \
        --n-sample     "${N_SAMPLE}" \
        --max-episode  "${MAX_EPISODE}" \
        --start-idle-seconds "${START_IDLE_SECONDS}"

    n_h5=$(ls "${DATASET_DIR}/${TASK_NAME}"/data_*.hdf5 2>/dev/null | wc -l)
    echo "[1/4] HDF5 episodes: ${n_h5}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — HDF5 → LeRobot
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "============================================================"
    echo "[2/4] converting → ${LEROBOT_DIR}"
    echo "============================================================"
    rm -rf "${LEROBOT_DIR}"
    ${CONVERT_PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${DATASET_DIR}/${TASK_NAME}" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps     10 \
        --top-k   "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec  "${VCODEC}" \
        --oracle-mode \
        --use-episode-instruction
fi

# ---------------------------------------------------------------------------
# Stage 3 — openpi train_pytorch.py
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "============================================================"
    echo "[3/4] training Pi0  (${TRAIN_STEPS} steps, batch=${BATCH_SIZE}, gpus=${NUM_GPUS})"
    echo "============================================================"

    if [[ -z "${OPENPI_PI0_JAX_WEIGHT:-}" || -z "${OPENPI_PI0_PYTORCH_WEIGHT:-}" ]]; then
        echo "[err] OPENPI_PI0_JAX_WEIGHT and OPENPI_PI0_PYTORCH_WEIGHT must be set."
        echo "      Point them at your local Pi0 base checkpoints."
        exit 1
    fi

    # The lerobot LeRobotDatasetMetadata loader resolves the dataset root
    # from HF_LEROBOT_HOME (default `~/.cache/huggingface/lerobot`) when the
    # openpi data_loader calls it without an explicit `root=`. Symlink our
    # local LeRobot dir into that exact path so the metadata pull skips the
    # HF API and uses the on-disk dataset.
    # `ln -sfn` (not bare `ln -s`) so a stale link from a previous run that
    # points at a now-missing/incomplete dataset dir is replaced rather than
    # silently kept — a dangling link sends the loader to the HF hub (404).
    LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${REPO_ID}"
    mkdir -p "$(dirname "${LEROBOT_CACHE}")"
    ln -sfn "${LEROBOT_DIR}" "${LEROBOT_CACHE}"
    echo "[3/4] symlinked ${LEROBOT_DIR} → ${LEROBOT_CACHE}"

    resume_flag=()
    if [[ "${RESUME}" == "1" ]]; then resume_flag+=(--resume); fi

    if (( NUM_GPUS > 1 )); then
        LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
    else
        LAUNCHER=()
    fi

    # Default wandb to offline: this box has no wandb credentials, and the
    # pi0 config leaves wandb_enabled=True, so an online init would abort the
    # run with "api_key not configured (no-tty)". Override WANDB_MODE=online
    # (after `wandb login`) if you want live logging.
    WANDB_MODE="${WANDB_MODE:-offline}"

    env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
        "WANDB_MODE=${WANDB_MODE}" \
        "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT}" \
        "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT}" \
        "OPENPI_FREEZE_PALIGEMMA=${FREEZE_PALIGEMMA}" \
        uv --project "${OPENPI_ROOT}" run \
        "${LAUNCHER[@]}" python "${OPENPI_ROOT}/scripts/train_pytorch.py" \
        "${POLICY_CONFIG}" \
        --exp_name "${EXP_NAME}" \
        --num_train_steps "${TRAIN_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --save_interval "${SAVE_INTERVAL}" \
        --checkpoint_base_dir "${OUTPUT_DIR}" \
        "${resume_flag[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 4 — eval. Delegates to sh/eval_pi05_audio.sh, which loads the openpi
# checkpoint locally via `AudioAwarePi05Policy` and runs the audio-aware
# evaluation loop in `src/eval/eval_smolvla_audio.evaluate_episode`. My
# take_out_microwave_food branch in that function is task-name-keyed so it
# works for both SmolVLA and Pi0 evaluation.
# ---------------------------------------------------------------------------
if [[ "${DO_EVAL}" == "1" ]]; then
    if [[ ! -d "${POLICY_DIR}" ]]; then
        echo "[err] POLICY_DIR not found: ${POLICY_DIR}"
        echo "      Set POLICY_DIR=/path/to/openpi/checkpoint if your run lives elsewhere."
        exit 1
    fi

    REPO_ROOT="${REPO_ROOT}" \
    OPENPI_ROOT="${OPENPI_ROOT}" \
    UV_CACHE_DIR="${UV_CACHE_DIR}" \
    TASK_NAME="${TASK_NAME}" \
    TAXONOMY="${TAXONOMY}" \
    LOCAL=1 \
    POLICY_CONFIG="${POLICY_CONFIG}" \
    POLICY_DIR="${POLICY_DIR}" \
    EVAL_MODE="${EVAL_MODE}" \
    EVAL_N="${EVAL_N}" \
    EVAL_MAX_LEN="${EVAL_MAX_LEN}" \
    EVAL_HORIZON="${EVAL_HORIZON}" \
    EVAL_DIR="${EVAL_DIR}" \
    EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO}" \
    EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD}" \
    AUDIO_CONFIG="${AUDIO_CONFIG}" \
    SLED_CKPT="${SLED_CKPT}" \
    TOP_K="${TOP_K}" \
    bash "${REPO_ROOT}/sh/eval_pi05_audio.sh"
fi

echo "============================================================"
echo "[done] pi0 pipeline for ${TASK_NAME}"
echo "  trajectories : ${DATASET_DIR}/${TASK_NAME}"
echo "  LeRobot      : ${LEROBOT_DIR}"
echo "  ckpts        : ${OUTPUT_DIR}/${EXP_NAME}/checkpoints/"
echo "  eval         : ${EVAL_DIR}"
echo "============================================================"

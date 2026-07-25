#!/usr/bin/env bash
# End-to-end audio-conditioned SmolVLA pipeline.
#
# Stages (each toggleable with DO_*=0/1):
#   0. install missing deps into the `vlabench` env  (one-time)
#   1. generate trajectories with binaural audio + SLED
#   2. convert HDF5 → LeRobot dataset (vlabench_unified-compatible format)
#   3. fine-tune the audio-aware SmolVLA
#
# Example presets:
#
#   # smoke test (≈ 1 min once the checkpoint is cached)
#   BATCH_SIZE=2 TRAIN_STEPS=5 WARMUP_STEPS=1 \
#       DO_GENERATE=0 DO_INSTALL=0 bash sh/test_smolvla_audio.sh
#
#   # quick sanity run ("does audio actually learn" — ≈ 3h on RTX 5090)
#   N_SAMPLE=150 TRAIN_STEPS=5000 BATCH_SIZE=32 NUM_WORKERS=8 \
#       bash sh/test_smolvla_audio.sh
#
#   # full training (≈ 10h on RTX 5090)
#   N_SAMPLE=400 TRAIN_STEPS=15000 BATCH_SIZE=32 NUM_WORKERS=8 \
#       WARMUP_STEPS=500 SAVE_EVERY=2000 \
#       bash sh/test_smolvla_audio.sh
#
#   # reuse existing trajectories, re-convert + re-train only
#   DO_GENERATE=0 bash sh/test_smolvla_audio.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/home/rllab/Desktop/AVLABench"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_v5}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_v5_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_audio}"

AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"

CONDA_ENV="${CONDA_ENV:-vlabench}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
REPO_ID="${REPO_ID:-local/avla_select_radio}"

# ---------------------------------------------------------------------------
# Stage toggles (set to 0 to skip a stage)
# ---------------------------------------------------------------------------
DO_INSTALL="${DO_INSTALL:-1}"
DO_GENERATE="${DO_GENERATE:-1}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_TRAIN="${DO_TRAIN:-1}"

# ---------------------------------------------------------------------------
# Stage 1 knobs
# ---------------------------------------------------------------------------
N_SAMPLE="${N_SAMPLE:-200}"       # raw trajectories to attempt (~40-60% succeed)

# ---------------------------------------------------------------------------
# Stage 2 knobs
# ---------------------------------------------------------------------------
TOP_K="${TOP_K:-3}"               # SLED events kept per frame
VCODEC="${VCODEC:-h264}"          # try "libsvtav1" if your ffmpeg has it
IMAGE_SIZE="${IMAGE_SIZE:-224}"   # matches vlabench_unified (224×224)

# ---------------------------------------------------------------------------
# Stage 3 knobs
# ---------------------------------------------------------------------------
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"    # 32 fits on RTX 5090 with frozen VLM
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"                  # action expert — conservative (preserve pretrained)
AUDIO_LR="${AUDIO_LR:-1e-4}"      # audio modules — faster (random init)
# Warmup defaults to 5% of TRAIN_STEPS, capped at 500. Override with WARMUP_STEPS=N.
_warmup_default=$(( TRAIN_STEPS / 20 ))
if   (( _warmup_default > 500 )); then _warmup_default=500
elif (( _warmup_default < 1   )); then _warmup_default=1
fi
WARMUP_STEPS="${WARMUP_STEPS:-${_warmup_default}}"
# Save every 10% of training, or every TRAIN_STEPS if < 10 (smoke).
_save_default=$(( TRAIN_STEPS / 10 ))
if (( _save_default < 1 )); then _save_default="${TRAIN_STEPS}"; fi
SAVE_EVERY="${SAVE_EVERY:-${_save_default}}"
LOG_EVERY="${LOG_EVERY:-20}"
TRAIN_AUDIO_ONLY="${TRAIN_AUDIO_ONLY:-0}"

# ---------------------------------------------------------------------------
PY="conda run -n ${CONDA_ENV} --no-capture-output python"
cd "${REPO_ROOT}"

cat <<EOF
[paths]   repo=${REPO_ROOT}
          dataset=${DATASET_DIR}
          lerobot_dataset=${LEROBOT_DIR}
          output=${OUTPUT_DIR}
[stages]  install=${DO_INSTALL} generate=${DO_GENERATE} convert=${DO_CONVERT} train=${DO_TRAIN}
[stage 3] steps=${TRAIN_STEPS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}
          lr=${LR} audio_lr=${AUDIO_LR} warmup=${WARMUP_STEPS}
          save_every=${SAVE_EVERY} log_every=${LOG_EVERY} audio_only=${TRAIN_AUDIO_ONLY}
EOF

# ---------------------------------------------------------------------------
# Stage 0 — install dependencies (idempotent; safe to re-run)
# ---------------------------------------------------------------------------
if [[ "${DO_INSTALL}" == "1" ]]; then
    echo "============================================================"
    echo "[0/3] installing deps into env=${CONDA_ENV}"
    echo "============================================================"
    # transformers < 5.0 so lerobot 0.4.4's GR00T import doesn't crash
    conda run -n "${CONDA_ENV}" pip install \
        "transformers>=4.57.1,<5.0.0" accelerate num2words
fi

# ---------------------------------------------------------------------------
# Stage 1 — generate trajectories with binaural audio + SLED
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    echo "============================================================"
    echo "[1/3] generating up to ${N_SAMPLE} trajectories"
    echo "============================================================"
    ${PY} scripts/trajectory_generation.py \
        --task-name    select_radio \
        --audio-config "${AUDIO_CONFIG}" \
        --sled-ckpt    "${SLED_CKPT}" \
        --save-dir     "${DATASET_DIR}" \
        --n-sample     "${N_SAMPLE}"

    n_h5=$(ls "${DATASET_DIR}/select_radio"/data_*.hdf5 2>/dev/null | wc -l)
    echo "[1/3] HDF5 episodes in ${DATASET_DIR}/select_radio: ${n_h5}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — convert HDF5 → LeRobot dataset (vlabench_unified schema)
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "============================================================"
    echo "[2/3] converting → ${LEROBOT_DIR}"
    echo "============================================================"
    rm -rf "${LEROBOT_DIR}"
    ${PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${DATASET_DIR}/select_radio" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps     10 \
        --top-k   "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec  "${VCODEC}"
fi

# ---------------------------------------------------------------------------
# Stage 3 — fine-tune the audio-aware SmolVLA
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "============================================================"
    echo "[3/3] training (${TRAIN_STEPS} steps, batch=${BATCH_SIZE})"
    echo "============================================================"
    rm -rf "${OUTPUT_DIR}"
    extra_args=()
    if [[ "${TRAIN_AUDIO_ONLY}" == "1" ]]; then
        extra_args+=(--train-audio-only)
    fi
    ${PY} src/training/train_smolvla_audio.py \
        --dataset-root "${LEROBOT_DIR}" \
        --repo-id      "${REPO_ID}" \
        --pretrained   "${PRETRAINED}" \
        --taxonomy     "${TAXONOMY}" \
        --output-dir   "${OUTPUT_DIR}" \
        --batch-size   "${BATCH_SIZE}" \
        --steps        "${TRAIN_STEPS}" \
        --lr           "${LR}" \
        --audio-lr     "${AUDIO_LR}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --num-workers  "${NUM_WORKERS}" \
        --save-every   "${SAVE_EVERY}" \
        --log-every    "${LOG_EVERY}" \
        "${extra_args[@]}"
fi

echo "============================================================"
echo "[done] all enabled stages finished"
echo "  raw trajectories : ${DATASET_DIR}/select_radio"
echo "  LeRobot dataset  : ${LEROBOT_DIR}"
echo "  checkpoints      : ${OUTPUT_DIR}"
echo "============================================================"

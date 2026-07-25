#!/usr/bin/env bash
# Evaluate a trained AudioAwareSmolVLA checkpoint on VLABench select_radio.
#
# Examples:
#   # default (latest ckpt in outputs/smolvla_audio/, 20 episodes, no video)
#   bash sh/eval_smolvla_audio.sh
#
#   # specific ckpt + save videos
#   CKPT=outputs/smolvla_audio/ckpt_step0005000.pt SAVE_VIDEO=1 N=10 \
#       bash sh/eval_smolvla_audio.sh
#
#   # comparison sweep across checkpoints
#   for c in outputs/smolvla_audio/ckpt_step000{2,4,6,8,10}000.pt; do
#       CKPT=$c SAVE_DIR=outputs/eval_$(basename $c .pt) \
#           bash sh/eval_smolvla_audio.sh
#   done

set -euo pipefail

REPO_ROOT="/home/rllab/Desktop/AVLABench"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-vlabench}"

# Default to the most recent ckpt in outputs/smolvla_audio/
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/outputs/smolvla_audio}"
CKPT="${CKPT:-$(ls -t ${CKPT_DIR}/ckpt_step*.pt 2>/dev/null | head -1)}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "[err] no ckpt found. Set CKPT=... or place ckpt_step*.pt in ${CKPT_DIR}"
    exit 1
fi

PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"

SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/eval_smolvla_audio}"
N="${N:-20}"
MAX_LEN="${MAX_LEN:-200}"
HORIZON="${HORIZON:-5}"
WARMUP_SECONDS="${WARMUP_SECONDS:-3.0}"
SEED="${SEED:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"
SAVE_AUDIO_LOG="${SAVE_AUDIO_LOG:-0}"

extra=()
if [[ "${SAVE_VIDEO}"     == "1" ]]; then extra+=(--save-video);     fi
if [[ "${SAVE_AUDIO}"     == "1" ]]; then extra+=(--save-audio);     fi
if [[ "${SAVE_AUDIO_LOG}" == "1" ]]; then extra+=(--save-audio-log); fi

echo "============================================================"
echo "[eval] ckpt        = ${CKPT}"
echo "       n_episodes  = ${N}"
echo "       max_length  = ${MAX_LEN}"
echo "       horizon     = ${HORIZON}"
echo "       save_dir    = ${SAVE_DIR}"
echo "       save_video  = ${SAVE_VIDEO}"
echo "============================================================"

conda run -n "${CONDA_ENV}" --no-capture-output python \
    src/eval/eval_smolvla_audio.py \
        --ckpt              "${CKPT}" \
        --pretrained        "${PRETRAINED}" \
        --taxonomy          "${TAXONOMY}" \
        --audio-config      "${AUDIO_CONFIG}" \
        --sled-ckpt         "${SLED_CKPT}" \
        --n-episodes        "${N}" \
        --max-episode-length "${MAX_LEN}" \
        --horizon           "${HORIZON}" \
        --warmup-seconds    "${WARMUP_SECONDS}" \
        --seed              "${SEED}" \
        --save-dir          "${SAVE_DIR}" \
        "${extra[@]}"

#!/bin/bash
# KISTI 스모크 테스트: 2×A100, batch16, 20 steps — 환경/런처/데이터 로딩 검증용
#   ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/smoke_pi0_kisti.slurm'
#SBATCH --job-name=pi0-smoke
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --comment pytorch
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

set -euo pipefail
export SCRATCH=/scratch/x3445a03
export PATH="${SCRATCH}/bin:${SCRATCH}/ffmpeg7/bin:${PATH}"
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/uv-python"
export XDG_DATA_HOME="${SCRATCH}/.local/share"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export LD_LIBRARY_PATH="${SCRATCH}/ffmpeg7/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_LEROBOT_HOME="${SCRATCH}/.cache/huggingface/lerobot"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${SCRATCH}/AVLABench"; mkdir -p logs
echo "Host $(hostname)  Start $(date)"; nvidia-smi -L || true

DO_NORM=1 DO_TRAIN=1 \
NUM_GPUS=2 BATCH_SIZE=16 NUM_WORKERS=4 \
TRAIN_STEPS=20 SAVE_INTERVAL=20 \
PEAK_LR=2e-4 WARMUP_STEPS=5 DECAY_STEPS=20 \
VLM_LORA=1 FREEZE_PALIGEMMA=0 \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
OUTPUT_DIR="${SCRATCH}/AVLABench/outputs/smoke" \
OPENPI_PI0_JAX_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params" \
OPENPI_PI0_PYTORCH_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "=== SMOKE_DONE $(date) ==="

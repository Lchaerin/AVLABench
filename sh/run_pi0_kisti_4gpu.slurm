#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: pi0 find_hidden, 4×A100 80G, global batch 128, 14k steps(~20ep), LoRA
#   제출: ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_pi0_kisti_4gpu.slurm'
#   1장 버전(run_pi0_kisti.slurm)의 "더 철저한" 버전 — 20에폭. 출력 디렉토리 분리.
#   4장은 한 노드에 4장 동시에 비어야 시작 → 만석이면 오래 PENDING 가능.
#   lr 2e-4 = sqrt 스케일(base 1e-4 @batch32 → batch128). DDP(torchrun) 경로.
# ===========================================================================
#SBATCH --job-name=pi0-fh-4gpu
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32             # a100nv_8 상한: GPU당 8코어 ×4
#SBATCH --time=1-06:00:00              # 30h 상한(실제 ~21h 예상, 14k에서 자동 종료)
#SBATCH --comment pytorch
#SBATCH --output=logs/pi0_4gpu_%j.out
#SBATCH --error=logs/pi0_4gpu_%j.err

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

cd "${SCRATCH}/AVLABench"
mkdir -p logs
echo "Host $(hostname)  Start $(date)"
nvidia-smi -L || true

# ---- 하이퍼파라미터 (4×A100 / batch 128 / 14k≈20ep / lr 2e-4) -------------
DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 \
NUM_GPUS=4 BATCH_SIZE="${BATCH_SIZE:-128}" NUM_WORKERS=6 \
TRAIN_STEPS="${TRAIN_STEPS:-14000}" SAVE_INTERVAL="${SAVE_INTERVAL:-2000}" \
PEAK_LR="${PEAK_LR:-2e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2000}" DECAY_STEPS="${DECAY_STEPS:-14000}" \
VLM_LORA=1 FREEZE_PALIGEMMA=0 LORA_RANK=16 LORA_ALPHA=16 \
RESUME="${RESUME:-0}" \
EXP_NAME=pi0_find_hidden_lora_kisti_4gpu \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
OUTPUT_DIR="${SCRATCH}/AVLABench/outputs/pi0_find_hidden_lora_kisti_4gpu" \
OPENPI_PI0_JAX_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params" \
OPENPI_PI0_PYTORCH_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "End $(date)"

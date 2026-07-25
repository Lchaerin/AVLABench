#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: pi0 find_hidden, 1×A100 80G, batch 32, 14k steps(~5ep), LoRA
#   제출:  ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_pi0_kisti.slurm'
#   설계: 1장은 4장보다 훨씬 빨리 스케줄됨(파티션 만석). QOS aanv8은 GPU 112까지 허용 =
#         4장 금지는 정책이 아니라 가용성 문제. batch 32 = openpi config 네이티브 기본값
#         → 80GB 단일 GPU에 맞고 lr 1e-4 그대로가 가장 검증된 설정(성공률 우선).
#   시간: batch32×14k=448k샘플(원본 320k의 1.4배, ~5.1ep). step 6s여도 ~23h로 24h 내 완주.
#   더 돌리려면(느리게 끝나 24h 초과 시): sbatch --export=ALL,RESUME=1,DO_NORM=0 sh/run_pi0_kisti.slurm
# ===========================================================================
#SBATCH --job-name=pi0-findhidden
#SBATCH --partition=amd_a100nv_8       # A100 80G
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1                    # 1장 (batch 32가 80GB에 들어감)
#SBATCH --cpus-per-task=8              # a100nv_8 상한: GPU당 8코어
#SBATCH --time=24:00:00               # 24h 이내 완료 목표
#SBATCH --comment pytorch
#SBATCH --output=logs/pi0_%j.out
#SBATCH --error=logs/pi0_%j.err

set -euo pipefail

# ---- 홈 quota 회피: 모든 캐시를 scratch 로 --------------------------------
export SCRATCH=/scratch/x3445a03
export PATH="${SCRATCH}/bin:${SCRATCH}/ffmpeg7/bin:${PATH}"   # uv + ffmpeg7
export UV_CACHE_DIR="${SCRATCH}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${SCRATCH}/uv-python"  # uv 관리형 python (홈 quota 회피)
export XDG_DATA_HOME="${SCRATCH}/.local/share"       # uv data dir (홈 quota 회피)
export XDG_CACHE_HOME="${SCRATCH}/.cache"
# av(PyAV)가 소스빌드돼 conda-forge ffmpeg7에 링크됨 → 런타임에도 필요
export LD_LIBRARY_PATH="${SCRATCH}/ffmpeg7/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_LEROBOT_HOME="${SCRATCH}/.cache/huggingface/lerobot"
export HF_HUB_OFFLINE=1                               # compute 노드 오프라인
export TRANSFORMERS_OFFLINE=1

cd "${SCRATCH}/AVLABench"
mkdir -p logs
echo "Host $(hostname)  Start $(date)"
nvidia-smi -L || true

# ---- 하이퍼파라미터 (1×A100 / batch 32 / 14k≈5.1ep / lr 1e-4 = config 기본) -
DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 \
NUM_GPUS=1 BATCH_SIZE="${BATCH_SIZE:-32}" NUM_WORKERS=8 \
TRAIN_STEPS="${TRAIN_STEPS:-14000}" SAVE_INTERVAL="${SAVE_INTERVAL:-2000}" \
PEAK_LR="${PEAK_LR:-1e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2000}" DECAY_STEPS="${DECAY_STEPS:-14000}" \
VLM_LORA=1 FREEZE_PALIGEMMA=0 LORA_RANK=16 LORA_ALPHA=16 \
RESUME="${RESUME:-0}" \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
OUTPUT_DIR="${SCRATCH}/AVLABench/outputs/pi0_find_hidden_lora_kisti" \
OPENPI_PI0_JAX_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params" \
OPENPI_PI0_PYTORCH_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "End $(date)"

#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: pi0 find_hidden, 1xA100 80G, batch 32, 14k steps, LoRA
#   + TWO-PHASE FINE-TUNE: steps 0-8399 as usual (PaliGemma LoRA + action
#     expert), then at step 8400 (60%) the whole PaliGemma tower freezes —
#     LoRA adapters included — and only the action expert + heads keep training.
#
#   제출: ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_pi0_kisti_stagedfreeze.slurm'
#
#   기존 모델 (오디오 브리지 미적용). 데이터셋은 v2 880ep의 real-state 변환본
#   (observation.state = 실측 EE 자세; 이전 KISTI 런들은 state==action 이었음).
#   전환 로직은 src/staged_freeze/ 에 있고 train_pytorch.py 는 두 줄만 호출합니다.
# ===========================================================================
#SBATCH --job-name=pi0-fh-sfreeze
#SBATCH --partition=amd_a100nv_8       # A100 80G
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8              # a100nv_8 상한: GPU당 8코어
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH --output=logs/pi0_sfreeze_%j.out
#SBATCH --error=logs/pi0_sfreeze_%j.err

set -euo pipefail

# ---- 홈 quota 회피: 모든 캐시를 scratch 로 --------------------------------
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

# 전환 로직이 실제로 import 되고 전환 스텝이 맞는지 잡 시작 시점에 확인.
# 시스템 python3 에는 torch 가 없으므로 반드시 openpi venv 로 확인할 것.
OPENPI_FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.6}" \
"${SCRATCH}/AVLABench/third_party/openpi/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${SCRATCH}/AVLABench')
from src.staged_freeze import config_from_env
c = config_from_env(${TRAIN_STEPS:-14000})
assert c is not None, 'staged freeze not armed'
print(f'[staged-freeze] preflight OK: switch at step {c.switch_step}/${TRAIN_STEPS:-14000}')
" || { echo '[err] src/staged_freeze preflight failed'; exit 1; }

# ---- 하이퍼파라미터 (1×A100 / batch 32 / 14k / lr 1e-4) -------------------
# FREEZE_LLM_AT_FRAC=0.6 -> 14000 * 0.6 = step 8400 에서 전환
DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 \
NUM_GPUS=1 BATCH_SIZE="${BATCH_SIZE:-32}" NUM_WORKERS=8 \
TRAIN_STEPS="${TRAIN_STEPS:-14000}" SAVE_INTERVAL="${SAVE_INTERVAL:-2000}" \
PEAK_LR="${PEAK_LR:-1e-4}" WARMUP_STEPS="${WARMUP_STEPS:-2000}" DECAY_STEPS="${DECAY_STEPS:-14000}" \
VLM_LORA=1 FREEZE_PALIGEMMA=0 LORA_RANK=16 LORA_ALPHA=16 \
FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.6}" \
FREEZE_LLM_AUDIO_HEADS="${FREEZE_LLM_AUDIO_HEADS:-0}" \
RESUME="${RESUME:-0}" \
EXP_NAME=pi0_find_hidden_kisti_stagedfreeze \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
LEROBOT_DIR="${SCRATCH}/AVLABench/dataset_find_hidden_lerobot_v2_realstate" \
OUTPUT_DIR="${SCRATCH}/AVLABench/outputs/pi0_find_hidden_kisti_stagedfreeze" \
OPENPI_PI0_JAX_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params" \
OPENPI_PI0_PYTORCH_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "End $(date)"

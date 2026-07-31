#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: pi0 select_radio (SELD-VLA slots_uv), 1xA100 80G
#   + TWO-PHASE FINE-TUNE: steps 0-13,999 as usual (PaliGemma LoRA + action
#     expert), then at step 14,000 (70%) the whole PaliGemma tower freezes —
#     LoRA adapters included — and only the action expert + heads keep training.
#
#   제출: ssh neuron 'cd /scratch/x3445a03/AVLABench && sbatch sh/run_pi0_kisti_radio_stagedfreeze.slurm'
#
#   하이퍼파라미터는 로컬 성공 런(outputs/eval_seld_uv_select_radio_only,
#   success 0.690 @ step 20000)과 동일:
#     batch 16 / 20,000 steps / decay 20,000 / save 5,000 / LoRA r16 a16
#     dropout 0.05 targets attn,mlp / peak_lr = openpi config 기본값(미지정)
#   유일한 차이가 마지막 30% 의 LLM freeze 이므로 통제된 비교가 됩니다.
# ===========================================================================
#SBATCH --job-name=pi0-radio-sfreeze
#SBATCH --partition=amd_a100nv_8       # A100 80G
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8              # a100nv_8 상한: GPU당 8코어
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH --output=logs/pi0_radio_sfreeze_%j.out
#SBATCH --error=logs/pi0_radio_sfreeze_%j.err

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

TRAIN_STEPS="${TRAIN_STEPS:-20000}"
FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.7}"

# 전환 로직 preflight. 시스템 python3 에는 torch 가 없으므로 openpi venv 로.
OPENPI_FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC}" \
"${SCRATCH}/AVLABench/third_party/openpi/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${SCRATCH}/AVLABench')
from src.staged_freeze import config_from_env
c = config_from_env(${TRAIN_STEPS})
assert c is not None, 'staged freeze not armed'
print(f'[staged-freeze] preflight OK: switch at step {c.switch_step}/${TRAIN_STEPS}')
" || { echo '[err] src/staged_freeze preflight failed'; exit 1; }

# ---- 로컬 성공 런과 동일한 하이퍼파라미터 (gen/convert/eval 제외) ---------
DO_CONVERT=0 DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 DO_EVAL=0 \
NUM_GPUS=1 BATCH_SIZE="${BATCH_SIZE:-16}" \
TRAIN_STEPS="${TRAIN_STEPS}" DECAY_STEPS="${DECAY_STEPS:-20000}" \
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}" \
VLM_LORA=1 LORA_RANK=16 LORA_ALPHA=16 LORA_DROPOUT=0.05 LORA_TARGETS=attn,mlp \
FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC}" \
FREEZE_LLM_AUDIO_HEADS="${FREEZE_LLM_AUDIO_HEADS:-0}" \
RESUME="${RESUME:-0}" \
EXP_NAME=pi0_seld_uv_select_radio_kisti_stagedfreeze \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
LEROBOT_DIR="${SCRATCH}/AVLABench/dataset_seld_uv_select_radio_lerobot" \
OUTPUT_DIR="${SCRATCH}/AVLABench/outputs/pi0_seld_uv_select_radio_kisti_stagedfreeze" \
OPENPI_PI0_JAX_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params" \
OPENPI_PI0_PYTORCH_WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch" \
bash sh/train_pi0_seld_uv_select_radio.sh

echo "End $(date)"

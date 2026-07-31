#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: find_hidden_object_open, 2xA100 80G DDP, all 4 slots,
# staged freeze at 70%. **One script, two backbones** — pick with BACKBONE.
#
#   BACKBONE=pi05  (default)  pi0.5: adaRMS timestep injection + discrete state
#                             tokens. checkpoints/pi05_base_torch.
#   BACKBONE=pi0              pi0:   continuous state_proj, max_token_len 48.
#                             checkpoints/pi0_base_primitive_torch.
#
# The two runs differ in *nothing* but the backbone — same data, batch, lr,
# schedule, freeze point, seed, GPUs. That is deliberate: they are a controlled
# pi0-vs-pi0.5 comparison, so keep them in one file rather than two that drift.
#
#   1. data      : dataset_find_hidden_lerobot_framefix — frame-fixed
#                  (ROBOT_FRAME_POS) + real-state, {left,right} x {top,bottom}.
#                  4슬롯을 전부 학습해야 평가의 슬롯별 분해로 **고도각(위/아래
#                  서랍) 능력**을 볼 수 있다. top-only 로 좁히면 그 축이 사라진다.
#   2. schedule  : steps 0..69% train everything (PaliGemma LoRA + action
#                  expert + audio heads); at 70% the PaliGemma tower freezes
#                  (LoRA adapters included) and only the action expert + heads
#                  keep training. src/staged_freeze/.
#
#   제출: ssh neuron 'cd /scratch/x3445a03/AVLABench && \
#           sbatch sh/run_kisti_findhidden.slurm                              # pi0.5
#           sbatch --export=ALL,BACKBONE=pi0 sh/run_kisti_findhidden.slurm    # pi0
#   스모크: sbatch --time=00:40:00 --export=ALL,SMOKE=1 sh/run_kisti_findhidden.slurm
# ===========================================================================
#SBATCH --job-name=fh-sfreeze
#SBATCH --partition=amd_a100nv_8       # A100 80G
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16             # a100nv_8 상한: GPU당 8코어
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH --output=logs/fh_%j.out
#SBATCH --error=logs/fh_%j.err

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

LEROBOT_DIR="${SCRATCH}/AVLABench/dataset_find_hidden_lerobot_framefix"

# ---- backbone 선택 ---------------------------------------------------------
BACKBONE="${BACKBONE:-pi05}"
weight_env=()
case "${BACKBONE}" in
    pi05)
        POLICY_CONFIG=pi05_ft_vlabench_find_hidden_lora
        WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi05_base_torch"
        weight_env+=("OPENPI_PI05_PYTORCH_WEIGHT=${WEIGHT}")
        ;;
    pi0)
        POLICY_CONFIG=pi0_ft_vlabench_find_hidden_framefix_lora
        WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive_torch"
        weight_env+=("OPENPI_PI0_PYTORCH_WEIGHT=${WEIGHT}")
        weight_env+=("OPENPI_PI0_JAX_WEIGHT=${SCRATCH}/AVLABench/checkpoints/pi0_base_primitive/params")
        ;;
    *)
        echo "[err] BACKBONE must be pi0 or pi05, got '${BACKBONE}'"; exit 1 ;;
esac

# ---- 하이퍼파라미터 (backbone 무관 — 비교 가능하게 동일하게 둔다) ----------
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
    TRAIN_STEPS="${TRAIN_STEPS:-30}"; SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
    WARMUP_STEPS="${WARMUP_STEPS:-5}"; DECAY_STEPS="${DECAY_STEPS:-30}"
    FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.5}"   # 15스텝에서 DDP 전환 검증
    EXP_NAME="${EXP_NAME:-${BACKBONE}_fh_smoke}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/AVLABench/outputs/smoke_${BACKBONE}_fh}"
else
    # 10,000 × batch 64 = 640k 샘플 = 85,893프레임(4슬롯) 기준 7.5 epoch.
    # 같은 데이터에서 success 0.050 을 낸 로컬 런은 batch16×20k = 3.7 epoch,
    # loss 는 이미 0.008 로 수렴해 있었다 → 7.5 epoch 은 충분히 여유 있는 값.
    # 위로 못 늘리는 이유는 24h 벽: 6 s/step 이면 16.7h, 최악 8.5 s/step 이어도
    # 23.6h 로 완주한다. pi0/pi0.5 를 **같은 step 수**로 맞춰야 비교가 성립한다.
    TRAIN_STEPS="${TRAIN_STEPS:-10000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"   # 7000(=전환 지점)에 정확히 걸림
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
    DECAY_STEPS="${DECAY_STEPS:-${TRAIN_STEPS}}"
    FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.7}"
    EXP_NAME="${EXP_NAME:-${BACKBONE}_find_hidden_stagedfreeze}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/AVLABench/outputs/${BACKBONE}_find_hidden_stagedfreeze}"
fi
BATCH_SIZE="${BATCH_SIZE:-64}"          # 글로벌. 2GPU → 32/GPU
PEAK_LR="${PEAK_LR:-2e-4}"              # 1e-4 @batch16 에서 sqrt(4) 스케일

echo "[run] backbone=${BACKBONE} config=${POLICY_CONFIG} weights=${WEIGHT}"
echo "[run] steps=${TRAIN_STEPS} batch=${BATCH_SIZE} lr=${PEAK_LR} freeze_at=${FREEZE_LLM_AT_FRAC}"

# ---- preflight: 전환 스텝 + 가중치 + 데이터셋 ------------------------------
# 시스템 python3 에는 torch 가 없다. 반드시 openpi venv 로 확인할 것.
[[ -d "${LEROBOT_DIR}" ]] || { echo "[err] missing ${LEROBOT_DIR}"; exit 1; }
[[ -f "${WEIGHT}/model.safetensors" ]] || { echo "[err] missing ${WEIGHT}/model.safetensors"; exit 1; }

OPENPI_FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC}" \
"${SCRATCH}/AVLABench/third_party/openpi/.venv/bin/python" -c "
import sys; sys.path.insert(0, '${SCRATCH}/AVLABench')
from src.staged_freeze import config_from_env
c = config_from_env(${TRAIN_STEPS})
assert c is not None, 'staged freeze not armed'
print(f'[staged-freeze] preflight OK: switch at step {c.switch_step}/${TRAIN_STEPS}')
" || { echo '[err] src/staged_freeze preflight failed'; exit 1; }

env "${weight_env[@]}" \
DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 \
NUM_GPUS=2 BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS=8 \
TRAIN_STEPS="${TRAIN_STEPS}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
PEAK_LR="${PEAK_LR}" WARMUP_STEPS="${WARMUP_STEPS}" DECAY_STEPS="${DECAY_STEPS}" \
VLM_LORA=1 FREEZE_PALIGEMMA=0 LORA_RANK=16 LORA_ALPHA=16 \
FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC}" \
FREEZE_LLM_AUDIO_HEADS="${FREEZE_LLM_AUDIO_HEADS:-0}" \
RESUME="${RESUME:-0}" \
POLICY_CONFIG="${POLICY_CONFIG}" \
EXP_NAME="${EXP_NAME}" \
REPO_ID=local/avla_find_hidden_framefix \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
LEROBOT_DIR="${LEROBOT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "End $(date)"

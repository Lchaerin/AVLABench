#!/bin/bash
# ===========================================================================
# KISTI Neuron sbatch: find_hidden_object_open on the **roll-fixed v3** data,
# **single A100 80G**, staged freeze at 70%. One script, two backbones.
#
#   BACKBONE=pi05  (default)  pi0.5: adaRMS timestep injection + discrete state
#                             tokens. checkpoints/pi05_base_torch.
#   BACKBONE=pi0              pi0:   continuous state_proj, max_token_len 48.
#                             checkpoints/pi0_base_primitive_torch.
#
# Differences vs sh/run_kisti_findhidden.slurm (the 2xA100 framefix pair this
# replaces):
#
#   1. data   : dataset_find_hidden_v3_rxfix_lerobot (repo_id
#               local/avla_find_hidden_v3_rxfix). Same 880 episodes / 87,087
#               frames as v3, but roll is folded onto a single (-2pi, 0] branch.
#               v3 left roll on the Euler +-pi seam, which made 3.07% of
#               DeltaActions chunks demand an impossible ~2pi rotation and put
#               70% of the roll target energy in those chunks. See the
#               2026-07-31 section of KISTI_pi0_STATUS.md.
#               ⚠️ A checkpoint from this run MUST be evaluated with an eval
#               loop that also folds roll (src/eval/eval_smolvla_audio.py has
#               it). Mixing conventions is the same class of train/eval
#               mismatch as the base-frame bug.
#   2. GPUs   : 2 -> **1**. The 2-GPU queue wait was the bottleneck, not the
#               compute. Note the partition gives 8 cores per GPU, so a 1-GPU
#               job also drops from 16 to 8 dataloader cores — that is why the
#               step budget below is set conservatively.
#   3. batch  : 64 global on one GPU (was 32/GPU x 2 = 64 global). Global batch
#               is unchanged, so **PEAK_LR stays 2e-4** — no rescale is called
#               for. 64 is the practical A100-80G ceiling here: batch 32 fits in
#               32 GB, so 64 lands near ~55 GB and 96 would not clear 80 GB.
#   4. steps  : 6,000 (was 10,000 on 2 GPUs). Sizing, from the measured
#               single-A100 run 865268 (batch 32, 16 cores, 4.73 s/step):
#                 batch 64 doubles GPU work    -> ~9.5 s/step  -> 15.8 h
#                 halved dataloader cores adds -> ~13  s/step  -> 21.7 h
#               Both clear the 24 h wall. 6,000 x 64 = 384k samples = 4.4 epochs
#               over 87,087 frames; the local run that first scored 0.050 saw
#               320k. SAVE_INTERVAL=600 divides the 4,200 freeze step exactly,
#               so even a wall-clock kill leaves post-freeze checkpoints.
#   5. schedule: steps 0..69% train everything (PaliGemma LoRA + action expert
#               + audio heads); at 70% (step 4,200) the PaliGemma tower freezes,
#               LoRA adapters included, and only the action expert + heads keep
#               training. src/staged_freeze/.
#
# The two backbones differ in *nothing* but the backbone — same data, batch, lr,
# schedule, freeze point, GPUs. Keep them in one file so they cannot drift.
#
#   제출: ssh neuron 'cd /scratch/x3445a03/AVLABench && \
#           sbatch sh/run_kisti_findhidden_rxfix.slurm                              # pi0.5
#           sbatch --export=ALL,BACKBONE=pi0 sh/run_kisti_findhidden_rxfix.slurm    # pi0
#   스모크: sbatch --time=00:40:00 --export=ALL,SMOKE=1 sh/run_kisti_findhidden_rxfix.slurm
# ===========================================================================
#SBATCH --job-name=fh-rxfix
#SBATCH --partition=amd_a100nv_8       # A100 80G
#SBATCH --nodes=1 --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8              # a100nv_8 상한: GPU당 8코어
#SBATCH --time=24:00:00
#SBATCH --comment pytorch
#SBATCH --output=logs/fhrx_%j.out
#SBATCH --error=logs/fhrx_%j.err

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

LEROBOT_DIR="${SCRATCH}/AVLABench/dataset_find_hidden_v3_rxfix_lerobot"

# ---- backbone 선택 ---------------------------------------------------------
BACKBONE="${BACKBONE:-pi05}"
weight_env=()
case "${BACKBONE}" in
    pi05)
        POLICY_CONFIG=pi05_ft_vlabench_find_hidden_v3_rxfix_lora
        WEIGHT="${SCRATCH}/AVLABench/checkpoints/pi05_base_torch"
        weight_env+=("OPENPI_PI05_PYTORCH_WEIGHT=${WEIGHT}")
        ;;
    pi0)
        POLICY_CONFIG=pi0_ft_vlabench_find_hidden_v3_rxfix_lora
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
    FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.5}"   # 15스텝에서 전환 검증
    EXP_NAME="${EXP_NAME:-${BACKBONE}_fhrx_smoke}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/AVLABench/outputs/smoke_${BACKBONE}_fhrx}"
else
    TRAIN_STEPS="${TRAIN_STEPS:-6000}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-600}"    # 4200(=전환 지점)에 정확히 걸림
    WARMUP_STEPS="${WARMUP_STEPS:-600}"      # 10%, 기존 런과 동일 비율
    DECAY_STEPS="${DECAY_STEPS:-${TRAIN_STEPS}}"
    FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC:-0.7}"
    EXP_NAME="${EXP_NAME:-${BACKBONE}_find_hidden_rxfix_stagedfreeze}"
    OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/AVLABench/outputs/${BACKBONE}_find_hidden_rxfix_stagedfreeze}"
fi
BATCH_SIZE="${BATCH_SIZE:-64}"          # 단일 GPU라 글로벌 = per-GPU
PEAK_LR="${PEAK_LR:-2e-4}"              # 글로벌 배치가 64로 동일 → 기존 값 유지

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

# 학습 데이터가 roll-fold 규약으로 만들어졌는지 확인한다. 규약이 섞이면
# 평가에서 조용히 틀리므로, 여기서 하드 페일시키는 편이 낫다.
"${SCRATCH}/AVLABench/third_party/openpi/.venv/bin/python" -c "
import glob, sys
import numpy as np, pandas as pd
ps = sorted(glob.glob('${LEROBOT_DIR}/data/**/*.parquet', recursive=True))
assert ps, 'no parquet under ${LEROBOT_DIR}/data'
bad = 0
for p in ps[::40]:
    df = pd.read_parquet(p)
    for col in ('observation.state', 'action'):
        rx = np.stack(df[col].values)[:, 3]
        bad += int((rx > 0).sum())
assert bad == 0, f'{bad} frames still on the positive roll branch — this is NOT the rxfix dataset'
print('[roll-fold] preflight OK: no positive-branch roll in sampled episodes')
" || { echo '[err] roll-fold preflight failed'; exit 1; }

env "${weight_env[@]}" \
DO_NORM="${DO_NORM:-1}" DO_TRAIN=1 \
NUM_GPUS=1 BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS=8 \
TRAIN_STEPS="${TRAIN_STEPS}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
PEAK_LR="${PEAK_LR}" WARMUP_STEPS="${WARMUP_STEPS}" DECAY_STEPS="${DECAY_STEPS}" \
VLM_LORA=1 FREEZE_PALIGEMMA=0 LORA_RANK=16 LORA_ALPHA=16 \
FREEZE_LLM_AT_FRAC="${FREEZE_LLM_AT_FRAC}" \
FREEZE_LLM_AUDIO_HEADS="${FREEZE_LLM_AUDIO_HEADS:-0}" \
RESUME="${RESUME:-0}" \
POLICY_CONFIG="${POLICY_CONFIG}" \
EXP_NAME="${EXP_NAME}" \
REPO_ID=local/avla_find_hidden_v3_rxfix \
REPO_ROOT="${SCRATCH}/AVLABench" \
OPENPI_ROOT="${SCRATCH}/AVLABench/third_party/openpi" \
UV_CACHE_DIR="${UV_CACHE_DIR}" \
LEROBOT_DIR="${LEROBOT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash sh/train_pi0_find_hidden_kisti.sh

echo "End $(date)"

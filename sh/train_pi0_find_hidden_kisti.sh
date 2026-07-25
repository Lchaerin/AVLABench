#!/usr/bin/env bash
# ===========================================================================
# KISTI Neuron 전용: pi0 find_hidden 학습 (norm stats + train만; gen/convert/eval 없음)
# 원본 sh/train_pi0_find_hidden.sh 를 건드리지 않는 독립 스크립트.
#
# 원본 대비 달라진 점:
#   * 멀티GPU 런처 수정: torchrun 사용 시 뒤의 `python` 제거(torchrun이 인터프리터를
#     직접 띄우므로 `torchrun ... python train.py`는 오작동)
#   * PEAK_LR / WARMUP_STEPS 오버라이드 노출 (원본은 DECAY_STEPS만 노출)
#   * gen/convert/eval 단계 제거 — 이미 변환된 LeRobot 데이터를 그대로 사용
#   * 모든 캐시를 /scratch 로 (홈 quota 회피는 호출부/sbatch에서 export)
#
# 필수 env: OPENPI_PI0_JAX_WEIGHT, OPENPI_PI0_PYTORCH_WEIGHT
# ===========================================================================
set -euo pipefail

# ---- paths (scratch 기본값; sbatch에서 덮어씀) -----------------------------
REPO_ROOT="${REPO_ROOT:-/scratch/x3445a03/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/x3445a03/uv-cache}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_find_hidden_lerobot}"
REPO_ID="${REPO_ID:-local/avla_find_hidden}"

# ---- stage switches --------------------------------------------------------
DO_NORM="${DO_NORM:-1}"
DO_TRAIN="${DO_TRAIN:-1}"

# ---- training --------------------------------------------------------------
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_find_hidden_lora}"
EXP_NAME="${EXP_NAME:-pi0_find_hidden_lora_kisti}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_find_hidden_lora_kisti}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"      # 글로벌 배치. 4GPU면 32/GPU
NUM_WORKERS="${NUM_WORKERS:-8}"      # 프로세스당. 4GPU×8 = 32코어(a100nv_8 상한)
# 데이터셋 48,183 프레임 기준: batch128 × 15k = ~40 에폭(원본 6.6ep의 6배).
# 원본(batch16×20k=6.6ep)이 loss 0.008로 수렴했으므로 15k면 충분+여유. 50k(133ep)는 과함.
TRAIN_STEPS="${TRAIN_STEPS:-15000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"   # 6개 체크포인트 → best 선택용
RESUME="${RESUME:-0}"

# lr (batch 32→128, sqrt 스케일 권장 2e-4)
PEAK_LR="${PEAK_LR:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1500}"     # ~10% of 15k
DECAY_STEPS="${DECAY_STEPS:-15000}"      # =TRAIN_STEPS → cosine 완전 감쇠

# LoRA (PaliGemma)
VLM_LORA="${VLM_LORA:-1}"
FREEZE_PALIGEMMA="${FREEZE_PALIGEMMA:-0}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn,mlp}"

WANDB_MODE="${WANDB_MODE:-offline}"

cd "${REPO_ROOT}"

cat <<EOF
[kisti find-hidden] norm=${DO_NORM} train=${DO_TRAIN}
   lerobot   = ${LEROBOT_DIR}   repo_id=${REPO_ID}
   config    = ${POLICY_CONFIG}   output=${OUTPUT_DIR}
   gpus=${NUM_GPUS}  global_batch=${BATCH_SIZE}  per_gpu=$(( BATCH_SIZE / NUM_GPUS ))  workers=${NUM_WORKERS}
   steps=${TRAIN_STEPS}  save=${SAVE_INTERVAL}  resume=${RESUME}
   lr: peak=${PEAK_LR} warmup=${WARMUP_STEPS} decay=${DECAY_STEPS}
   lora=${VLM_LORA} (rank=${LORA_RANK} alpha=${LORA_ALPHA})  freeze_pg=${FREEZE_PALIGEMMA}
EOF

if [[ -z "${OPENPI_PI0_JAX_WEIGHT:-}" || -z "${OPENPI_PI0_PYTORCH_WEIGHT:-}" ]]; then
    echo "[err] OPENPI_PI0_JAX_WEIGHT / OPENPI_PI0_PYTORCH_WEIGHT 를 export 하세요."; exit 1
fi
[[ -d "${LEROBOT_DIR}" ]] || { echo "[err] missing ${LEROBOT_DIR}"; exit 1; }

# openpi 로더가 HF API를 건너뛰도록 로컬 데이터셋을 HF_LEROBOT_HOME 에 심링크
LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${REPO_ID}"
mkdir -p "$(dirname "${LEROBOT_CACHE}")"
ln -sfn "${LEROBOT_DIR}" "${LEROBOT_CACHE}"

# ---------------------------------------------------------------------------
# [1] norm stats
# ---------------------------------------------------------------------------
if [[ "${DO_NORM}" == "1" ]]; then
    echo "==================== [1] compute norm stats ===================="
    env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
        "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT}" \
        "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT}" \
        uv --project "${OPENPI_ROOT}" run python \
        "${OPENPI_ROOT}/scripts/compute_norm_stats.py" --config-name "${POLICY_CONFIG}"
fi

# ---------------------------------------------------------------------------
# [2] train (LoRA on PaliGemma, DDP)
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "==================== [2] train ${POLICY_CONFIG} ===================="

    # ---- 런처: 멀티GPU면 torchrun(뒤에 python 없음), 단일이면 python ----
    if (( NUM_GPUS > 1 )); then
        runner=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}"
                "${OPENPI_ROOT}/scripts/train_pytorch.py")
    else
        runner=(python "${OPENPI_ROOT}/scripts/train_pytorch.py")
    fi

    resume_flag=(); [[ "${RESUME}" == "1" ]] && resume_flag+=(--resume)
    lr_flags=(--lr_schedule.peak_lr "${PEAK_LR}"
              --lr_schedule.warmup_steps "${WARMUP_STEPS}"
              --lr_schedule.decay_steps "${DECAY_STEPS}")

    lora_env=()
    if [[ "${VLM_LORA}" == "1" ]]; then
        lora_env=(
            "OPENPI_PALIGEMMA_LORA=1"
            "OPENPI_LORA_RANK=${LORA_RANK}"
            "OPENPI_LORA_ALPHA=${LORA_ALPHA}"
            "OPENPI_LORA_DROPOUT=${LORA_DROPOUT}"
            "OPENPI_LORA_TARGETS=${LORA_TARGETS}"
        )
    fi

    env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
        "WANDB_MODE=${WANDB_MODE}" \
        "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT}" \
        "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT}" \
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
        "OPENPI_FREEZE_PALIGEMMA=${FREEZE_PALIGEMMA}" \
        "${lora_env[@]}" \
        uv --project "${OPENPI_ROOT}" run \
        "${runner[@]}" \
        "${POLICY_CONFIG}" \
        --exp_name "${EXP_NAME}" \
        --num_train_steps "${TRAIN_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --save_interval "${SAVE_INTERVAL}" \
        --checkpoint_base_dir "${OUTPUT_DIR}" \
        "${resume_flag[@]}" \
        "${lr_flags[@]}"
fi

echo "==================== [done] kisti find_hidden pi0 ===================="
echo "  checkpoints : ${OUTPUT_DIR}/${POLICY_CONFIG}/${EXP_NAME}"

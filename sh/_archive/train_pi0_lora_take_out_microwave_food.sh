#!/usr/bin/env bash
# LoRA fine-tuning of Pi0 (PaliGemma LLM) for take_out_microwave_food.
#
# Difference vs sh/train_pi0_take_out_microwave_food.sh
# -----------------------------------------------------
#   That script freezes ALL of PaliGemma and trains only the action expert +
#   heads. The LLM never adapts, which caps the visuomotor mapping the policy
#   can learn (eval success stayed ~0 even after the action loss plateaued by
#   ~10-20k steps). This script instead injects LoRA adapters (rank 16) into
#   the PaliGemma LLM attention+MLP so the language tower can adapt cheaply,
#   while keeping the optimizer state small enough for a single 32GB GPU.
#
#   LoRA is applied at runtime by train_pytorch.py (OPENPI_PALIGEMMA_LORA=1);
#   it freezes the paligemma base (vision tower + LLM) and trains the adapters
#   + action expert + projection/audio heads. The recipe is recorded to
#   lora_config.json in every checkpoint so eval re-injects it automatically
#   (no eval-side flags needed).
#
# Chosen hyperparameters (the "reasonable defaults")
# --------------------------------------------------
#   rank 16 / alpha 16 (scaling 1.0)   - openpi's own gemma_2b_lora recipe
#   dropout 0.05                       - light reg for the ~2k-demo dataset
#   targets attn+mlp                   - adapt both, matching gemma_2b_lora
#   peak_lr 1e-4, cosine to 1e-5       - higher than the 2.5e-5 frozen recipe;
#                                        LoRA tolerates/wants it (set in the
#                                        pi0_ft_..._lora TrainConfig)
#   batch 32                           - gradient checkpointing is already on,
#                                        so the deeper backward still fits; drop
#                                        to 16 via BATCH_SIZE=16 if you OOM.
#
# REQUIRED env (same Pi0 base weights as the frozen pipeline):
#   OPENPI_PI0_JAX_WEIGHT, OPENPI_PI0_PYTORCH_WEIGHT
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_take_out_microwave_food_oracle_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_lora_take_out_microwave_food}"
REPO_ID="${REPO_ID:-local/avla_take_out_microwave_food}"

POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_take_out_microwave_food_lora}"
EXP_NAME="${EXP_NAME:-pi0_lora_take_out_microwave_food}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUME="${RESUME:-0}"

# ---- LoRA knobs (read by train_pytorch.py via lora_pytorch.lora_request_from_env)
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn,mlp}"

# Match the training-time chime distribution used by the frozen pipeline.
export VLABENCH_MICROWAVE_MIN_DELAY_SEC="${VLABENCH_MICROWAVE_MIN_DELAY_SEC:-0.0}"
export VLABENCH_MICROWAVE_MAX_DELAY_SEC="${VLABENCH_MICROWAVE_MAX_DELAY_SEC:-20.0}"
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC="${VLABENCH_MICROWAVE_REACT_WINDOW_SEC:-15.0}"

cd "${REPO_ROOT}"

if [[ -z "${OPENPI_PI0_JAX_WEIGHT:-}" || -z "${OPENPI_PI0_PYTORCH_WEIGHT:-}" ]]; then
    echo "[err] OPENPI_PI0_JAX_WEIGHT and OPENPI_PI0_PYTORCH_WEIGHT must be set."
    exit 1
fi

# Symlink the local LeRobot dataset into HF_LEROBOT_HOME so the loader skips the
# HF API (same trick as the frozen pipeline).
LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${REPO_ID}"
mkdir -p "$(dirname "${LEROBOT_CACHE}")"
ln -sfn "${LEROBOT_DIR}" "${LEROBOT_CACHE}"

# Reuse the frozen config's normalization stats: the dataset + data transforms
# are identical, so the per-config norm_stats are byte-identical. openpi looks
# them up at assets/<config_name>/<asset_id>/norm_stats.json, so copy them under
# the LoRA config name if missing (otherwise it aborts asking for
# compute_norm_stats.py).
_ASSET_ID="local/avla_take_out_microwave_food"
_NS_SRC="${REPO_ROOT}/assets/pi0_ft_vlabench_take_out_microwave_food/${_ASSET_ID}/norm_stats.json"
_NS_DST="${REPO_ROOT}/assets/${POLICY_CONFIG}/${_ASSET_ID}/norm_stats.json"
if [[ ! -f "${_NS_DST}" && -f "${_NS_SRC}" ]]; then
    mkdir -p "$(dirname "${_NS_DST}")"
    cp "${_NS_SRC}" "${_NS_DST}"
    echo "[lora-train] reused norm_stats: ${_NS_SRC} -> ${_NS_DST}"
fi

resume_flag=()
if [[ "${RESUME}" == "1" ]]; then resume_flag+=(--resume); fi

if (( NUM_GPUS > 1 )); then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
else
    LAUNCHER=()
fi

WANDB_MODE="${WANDB_MODE:-offline}"

cat <<EOF
[lora-train] config=${POLICY_CONFIG} exp=${EXP_NAME}
             steps=${TRAIN_STEPS} batch=${BATCH_SIZE} gpus=${NUM_GPUS}
             lora rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGETS}
             output=${OUTPUT_DIR}
EOF

env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
    "WANDB_MODE=${WANDB_MODE}" \
    "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT}" \
    "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT}" \
    "OPENPI_PALIGEMMA_LORA=1" \
    "OPENPI_LORA_RANK=${LORA_RANK}" \
    "OPENPI_LORA_ALPHA=${LORA_ALPHA}" \
    "OPENPI_LORA_DROPOUT=${LORA_DROPOUT}" \
    "OPENPI_LORA_TARGETS=${LORA_TARGETS}" \
    uv --project "${OPENPI_ROOT}" run \
    "${LAUNCHER[@]}" python "${OPENPI_ROOT}/scripts/train_pytorch.py" \
    "${POLICY_CONFIG}" \
    --exp_name "${EXP_NAME}" \
    --num_train_steps "${TRAIN_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --save_interval "${SAVE_INTERVAL}" \
    --checkpoint_base_dir "${OUTPUT_DIR}" \
    "${resume_flag[@]}"

echo "[done] LoRA checkpoints -> ${OUTPUT_DIR}/${EXP_NAME}/checkpoints/"
echo "       eval with sh/eval_pi05_audio.sh (LoRA auto-detected via lora_config.json),"
echo "       POLICY_CONFIG=${POLICY_CONFIG}  POLICY_DIR=.../checkpoints/<step>  --audio-mode slots"

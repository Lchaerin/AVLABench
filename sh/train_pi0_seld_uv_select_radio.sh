#!/usr/bin/env bash
# SELD-VLA (pi0, SlotEncoder path) single-task pipeline for select_radio only:
#   press the button in front of the one radio that is making sound.
#
# Unlike sh/train_pi0_seld_uv_three_radio.sh, this trains on select_radio
# alone (no combine step across the 3 radio tasks). Dataset generation /
# balancing for select_radio is expected to already be done, e.g. via
# sh/gen_seld_uv_balance.sh, which tops up dataset_seld_uv/select_radio to a
# balanced left/middle/right distribution.
#
# REQUIRED env (same pi0 base weights as the other openpi pipelines):
#   OPENPI_PI0_JAX_WEIGHT, OPENPI_PI0_PYTORCH_WEIGHT
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
# Conversion MUST run in the openpi venv (lerobot 0.1.0) so it writes the v2.1
# LeRobot format the openpi training loader reads.
CONVERT_PY=(env "UV_CACHE_DIR=${UV_CACHE_DIR}" uv --project "${OPENPI_ROOT}" run python)

TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
REPO_ID="${REPO_ID:-local/avla_select_radio_seld_uv_only}"

# ---- data locations --------------------------------------------------------
SRC_DIR="${SRC_DIR:-${REPO_ROOT}/dataset_seld_uv/select_radio}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_seld_uv_select_radio_lerobot}"

# ---- stage switches --------------------------------------------------------
DO_CONVERT="${DO_CONVERT:-1}"
DO_NORM="${DO_NORM:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

# ---- conversion ------------------------------------------------------------
TOP_K="${TOP_K:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VCODEC="${VCODEC:-h264}"

# ---- training (openpi) -----------------------------------------------------
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_seld_uv_select_radio_lora}"
EXP_NAME="${EXP_NAME:-pi0_seld_uv_select_radio_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_seld_uv_select_radio_lora}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
# lr_schedule.decay_steps defaults to the config's built-in 60000 (see
# config.py). When TRAIN_STEPS is cut short of that (e.g. to fit a 24h
# wall-clock budget), pass DECAY_STEPS=TRAIN_STEPS so the cosine schedule
# actually anneals to decay_lr by the end of the run instead of stopping
# mid-decay at a still-high LR.
DECAY_STEPS="${DECAY_STEPS:-}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUME="${RESUME:-0}"
VLM_LORA="${VLM_LORA:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn,mlp}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ---- eval ------------------------------------------------------------------
EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_STEP="${EVAL_STEP:-}"        # checkpoint step to eval; default = latest

cd "${REPO_ROOT}"

cat <<EOF
[seld-uv-select-radio] convert=${DO_CONVERT} norm=${DO_NORM} train=${DO_TRAIN} eval=${DO_EVAL}
          src_dir  = ${SRC_DIR}
          lerobot  = ${LEROBOT_DIR}   repo_id=${REPO_ID}
          config   = ${POLICY_CONFIG}  output=${OUTPUT_DIR}
          steps=${TRAIN_STEPS} save_interval=${SAVE_INTERVAL} batch=${BATCH_SIZE} lora=${VLM_LORA}
EOF

# ---------------------------------------------------------------------------
# [1] convert select_radio HDF5 -> LeRobot (writes energy + uv columns)
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "==================== [1] convert -> LeRobot ===================="
    rm -rf "${LEROBOT_DIR}"
    "${CONVERT_PY[@]}" src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${SRC_DIR}" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps 10 \
        --top-k "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec "${VCODEC}" \
        --oracle-mode \
        --use-episode-instruction
fi

# Symlink the local LeRobot dataset into HF_LEROBOT_HOME so openpi's loader
# skips the HF API (same trick as the other openpi pipelines).
LEROBOT_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${REPO_ID}"
mkdir -p "$(dirname "${LEROBOT_CACHE}")"
ln -sfn "${LEROBOT_DIR}" "${LEROBOT_CACHE}"

# ---------------------------------------------------------------------------
# [2] compute normalization stats for the config (fresh config => required)
# ---------------------------------------------------------------------------
if [[ "${DO_NORM}" == "1" ]]; then
    echo "==================== [2] compute norm stats ===================="
    env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
        "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT:-}" \
        "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT:-}" \
        uv --project "${OPENPI_ROOT}" run python \
        "${OPENPI_ROOT}/scripts/compute_norm_stats.py" --config-name "${POLICY_CONFIG}"
fi

# ---------------------------------------------------------------------------
# [3] train pi0 (LoRA on PaliGemma by default)
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "==================== [3] train ${POLICY_CONFIG} ===================="
    if [[ -z "${OPENPI_PI0_JAX_WEIGHT:-}" || -z "${OPENPI_PI0_PYTORCH_WEIGHT:-}" ]]; then
        echo "[err] OPENPI_PI0_JAX_WEIGHT and OPENPI_PI0_PYTORCH_WEIGHT must be set."; exit 1
    fi
    resume_flag=(); [[ "${RESUME}" == "1" ]] && resume_flag+=(--resume)
    decay_flag=(); [[ -n "${DECAY_STEPS}" ]] && decay_flag+=(--lr_schedule.decay_steps "${DECAY_STEPS}")
    launcher=()
    (( NUM_GPUS > 1 )) && launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
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
        "${lora_env[@]}" \
        uv --project "${OPENPI_ROOT}" run \
        "${launcher[@]}" python "${OPENPI_ROOT}/scripts/train_pytorch.py" \
        "${POLICY_CONFIG}" \
        --exp_name "${EXP_NAME}" \
        --num_train_steps "${TRAIN_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --save_interval "${SAVE_INTERVAL}" \
        --checkpoint_base_dir "${OUTPUT_DIR}" \
        "${resume_flag[@]}" \
        "${decay_flag[@]}"
fi

# ---------------------------------------------------------------------------
# [4] eval (audio-mode slots_uv) on select_radio
# ---------------------------------------------------------------------------
# Eval-time oracle noise. Training data is CLEAN (conf=1.0, un-noised uv/
# energy/class), so eval defaults to clean too (all noise 0, conf 1.0) for
# train/eval parity. Override to probe robustness.
EVAL_NOISE_AZ_STD="${EVAL_NOISE_AZ_STD:-0}"
EVAL_NOISE_EL_STD="${EVAL_NOISE_EL_STD:-0}"
EVAL_NOISE_CONF_MIN="${EVAL_NOISE_CONF_MIN:-1.0}"
EVAL_NOISE_CONF_MAX="${EVAL_NOISE_CONF_MAX:-1.0}"
EVAL_NOISE_CLASS_FLIP_PROB="${EVAL_NOISE_CLASS_FLIP_PROB:-0}"
EVAL_NOISE_ENERGY_STD="${EVAL_NOISE_ENERGY_STD:-0}"

if [[ "${DO_EVAL}" == "1" ]]; then
    # openpi saves to  base_dir / POLICY_CONFIG / EXP_NAME / <step>
    CKPT_ROOT="${OUTPUT_DIR}/${POLICY_CONFIG}/${EXP_NAME}"
    if [[ -n "${EVAL_STEP}" ]]; then
        POLICY_DIR="${CKPT_ROOT}/${EVAL_STEP}"
    else
        POLICY_DIR="$(ls -dt ${CKPT_ROOT}/*/ 2>/dev/null | head -1)"
    fi
    POLICY_DIR="${POLICY_DIR%/}"
    if [[ -z "${POLICY_DIR}" || ! -d "${POLICY_DIR}" ]]; then
        echo "[err] no checkpoint under ${CKPT_ROOT}"; exit 1
    fi
    echo "[eval] using checkpoint ${POLICY_DIR}"
    echo "==================== [4] eval select_radio ===================="
    EVAL_DIR="${REPO_ROOT}/outputs/eval_seld_uv_select_radio_only" TASK_NAME="select_radio" TAXONOMY="${TAXONOMY}" \
    LOCAL=1 POLICY_CONFIG="${POLICY_CONFIG}" POLICY_DIR="${POLICY_DIR}" \
    OPENPI_ROOT="${OPENPI_ROOT}" AUDIO_MODE="slots_uv" EVAL_MODE="oracle" \
    EVAL_N="${EVAL_N}" EVAL_MAX_LEN="${EVAL_MAX_LEN}" EVAL_HORIZON="${EVAL_HORIZON}" \
    TOP_K="${TOP_K}" \
    NOISE_AZ_STD="${EVAL_NOISE_AZ_STD}" NOISE_EL_STD="${EVAL_NOISE_EL_STD}" \
    NOISE_CONF_MIN="${EVAL_NOISE_CONF_MIN}" NOISE_CONF_MAX="${EVAL_NOISE_CONF_MAX}" \
    NOISE_CLASS_FLIP_PROB="${EVAL_NOISE_CLASS_FLIP_PROB}" \
    NOISE_ENERGY_STD="${EVAL_NOISE_ENERGY_STD}" \
    bash sh/eval_pi05_audio.sh
fi

echo "==================== [done] SELD-VLA pi0 select_radio-only pipeline ===================="
echo "  lerobot     : ${LEROBOT_DIR}"
echo "  checkpoints : ${OUTPUT_DIR}/${POLICY_CONFIG}/${EXP_NAME}"

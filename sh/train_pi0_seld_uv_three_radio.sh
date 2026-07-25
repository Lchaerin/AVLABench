#!/usr/bin/env bash
# SELD-VLA (pi0, SlotEncoder path) end-to-end pipeline for the 3 radio tasks:
#   1) select_radio        — press the one radio that is making sound.
#   2) select_radio_two    — two radios play different sounds; press the one
#                            matching the requested sound class.
#   3) select_radio_silent — three radios, two sounding; press the quiet one.
#
# This is the oracle pipeline for the new audio format. Because SLED now also
# reports loudness/energy (and we project the DoA to image coordinates), the
# HDF5 + LeRobot datasets must be regenerated from scratch — extract_episode_gt
# now records per-source energy + camera FOV, and convert_hdf5_to_lerobot.py
# writes observation.audio.energy + observation.audio.uv columns.
#
# Audio reaches pi0 via audio_mode="slots_uv" (spec M4): class words + loudness
# go into the prompt text and (u,v)+energy+conf feed the continuous SlotEncoder
# token. Training uses the openpi backbone (LoRA on PaliGemma by default).
#
# REQUIRED env (same pi0 base weights as the other openpi pipelines):
#   OPENPI_PI0_JAX_WEIGHT, OPENPI_PI0_PYTORCH_WEIGHT
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
# Generation needs the VLABench sim (mujoco) → conda env.
PY="conda run -n ${CONDA_ENV} --no-capture-output python"
# Conversion MUST run in the openpi venv (lerobot 0.1.0) so it writes the v2.1
# LeRobot format the openpi training loader reads. Converting under conda's
# lerobot 0.4.x produces v3.0 (tasks.parquet) which openpi can't load → HF 404.
CONVERT_PY=(env "UV_CACHE_DIR=${UV_CACHE_DIR}" uv --project "${OPENPI_ROOT}" run python)

TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
REPO_ID="${REPO_ID:-local/avla_select_radio_seld_uv}"

# ---- data locations --------------------------------------------------------
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_seld_uv}"
ONE_SRC_DIR="${ONE_SRC_DIR:-${GEN_ROOT}/select_radio}"
TWO_SRC_DIR="${TWO_SRC_DIR:-${GEN_ROOT}/select_radio_two}"
SILENT_SRC_DIR="${SILENT_SRC_DIR:-${GEN_ROOT}/select_radio_silent}"
COMBINED_DATASET_DIR="${COMBINED_DATASET_DIR:-${REPO_ROOT}/dataset_seld_uv_mixed}"
COMBINED_TASK_NAME="${COMBINED_TASK_NAME:-select_radio_seld_uv}"
COMBINED_SRC_DIR="${COMBINED_DATASET_DIR}/${COMBINED_TASK_NAME}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_seld_uv_mixed_lerobot}"

# ---- stage switches --------------------------------------------------------
DO_GENERATE="${DO_GENERATE:-1}"
DO_COMBINE="${DO_COMBINE:-1}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_NORM="${DO_NORM:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

# ---- generation ------------------------------------------------------------
N_SAMPLE="${N_SAMPLE:-500}"
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"

# ---- conversion ------------------------------------------------------------
TOP_K="${TOP_K:-3}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
VCODEC="${VCODEC:-h264}"

# ---- training (openpi) -----------------------------------------------------
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_seld_uv_three_radio_lora}"
EXP_NAME="${EXP_NAME:-pi0_seld_uv_three_radio_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/pi0_seld_uv_three_radio_lora}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
RESUME="${RESUME:-0}"
VLM_LORA="${VLM_LORA:-1}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn,mlp}"
WANDB_MODE="${WANDB_MODE:-offline}"

# ---- eval ------------------------------------------------------------------
EVAL_TASKS="${EVAL_TASKS:-all}"   # all | one | two | silent
EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_STEP="${EVAL_STEP:-}"        # checkpoint step to eval; default = latest

cd "${REPO_ROOT}"

cat <<EOF
[seld-uv] gen=${DO_GENERATE} combine=${DO_COMBINE} convert=${DO_CONVERT} norm=${DO_NORM} train=${DO_TRAIN} eval=${DO_EVAL}
          gen_root = ${GEN_ROOT}
          lerobot  = ${LEROBOT_DIR}   repo_id=${REPO_ID}
          config   = ${POLICY_CONFIG}  output=${OUTPUT_DIR}
          steps=${TRAIN_STEPS} batch=${BATCH_SIZE} num_workers=${NUM_WORKERS} lora=${VLM_LORA}
EOF

# ---------------------------------------------------------------------------
# [1] generate oracle HDF5 for each task
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    for spec in "select_radio:${ONE_SRC_DIR}" \
                "select_radio_two:${TWO_SRC_DIR}" \
                "select_radio_silent:${SILENT_SRC_DIR}"; do
        task="${spec%%:*}"; dir="${spec##*:}"
        echo "==================== [1] generate ${task} ===================="
        parent="$(dirname "${dir}")"
        ${PY} scripts/trajectory_generation.py \
            --task-name "${task}" \
            --oracle-mode \
            --save-dir "${parent}" \
            --n-sample "${N_SAMPLE}" \
            --max-episode "${N_SAMPLE}" \
            --start-idle-seconds "${START_IDLE_SECONDS}" \
            --dataset-fps "${DATASET_FPS}"
    done
fi

# ---------------------------------------------------------------------------
# [2] combine the three tasks' HDF5 into one directory (per-episode
#     instructions preserved; one-radio prompt normalised to audio-conditioned)
# ---------------------------------------------------------------------------
if [[ "${DO_COMBINE}" == "1" ]]; then
    echo "==================== [2] combine HDF5 ===================="
    rm -rf "${COMBINED_SRC_DIR}"
    mkdir -p "${COMBINED_SRC_DIR}"
    ONE_SRC_DIR="${ONE_SRC_DIR}" TWO_SRC_DIR="${TWO_SRC_DIR}" \
    SILENT_SRC_DIR="${SILENT_SRC_DIR}" COMBINED_SRC_DIR="${COMBINED_SRC_DIR}" \
    ${PY} - <<'PY'
import os, shutil
from pathlib import Path
import h5py, numpy as np

sources = [
    ("one",    Path(os.environ["ONE_SRC_DIR"])),
    ("two",    Path(os.environ["TWO_SRC_DIR"])),
    ("silent", Path(os.environ["SILENT_SRC_DIR"])),
]
out = Path(os.environ["COMBINED_SRC_DIR"])
fixed_one = "primitive: Press the button in front of the radio that is making sound."
idx = 0; counts = {}
for label, src in sources:
    files = sorted(src.glob("data_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no data_*.hdf5 in {src}")
    counts[label] = 0
    for path in files:
        try:
            with h5py.File(path, "r") as f:
                _ = next(iter(f["data"].keys()))
        except Exception as exc:
            print(f"[warn] skip corrupt {label} {path.name}: {exc}"); continue
        dst = out / f"data_{idx:06d}.hdf5"
        if label == "one":
            shutil.copy2(path, dst)
            with h5py.File(dst, "r+") as f:
                ep = f["data"][next(iter(f["data"].keys()))]
                if "instruction" in ep:
                    del ep["instruction"]
                ep.create_dataset("instruction",
                                  data=np.array([fixed_one], dtype=h5py.string_dtype("utf-8")))
        else:
            try:
                os.link(path, dst)
            except OSError:
                shutil.copy2(path, dst)
        idx += 1; counts[label] += 1
print({"counts": counts, "total": idx, "out": str(out)})
PY
fi

# ---------------------------------------------------------------------------
# [3] convert combined HDF5 -> LeRobot (writes energy + uv columns)
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "==================== [3] convert -> LeRobot ===================="
    rm -rf "${LEROBOT_DIR}"
    "${CONVERT_PY[@]}" src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${COMBINED_SRC_DIR}" \
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
# [4] compute normalization stats for the config (fresh config => required)
# ---------------------------------------------------------------------------
if [[ "${DO_NORM}" == "1" ]]; then
    echo "==================== [4] compute norm stats ===================="
    env "UV_CACHE_DIR=${UV_CACHE_DIR}" \
        "OPENPI_PI0_JAX_WEIGHT=${OPENPI_PI0_JAX_WEIGHT:-}" \
        "OPENPI_PI0_PYTORCH_WEIGHT=${OPENPI_PI0_PYTORCH_WEIGHT:-}" \
        uv --project "${OPENPI_ROOT}" run python \
        "${OPENPI_ROOT}/scripts/compute_norm_stats.py" --config-name "${POLICY_CONFIG}"
fi

# ---------------------------------------------------------------------------
# [5] train pi0 (LoRA on PaliGemma by default)
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "==================== [5] train ${POLICY_CONFIG} ===================="
    if [[ -z "${OPENPI_PI0_JAX_WEIGHT:-}" || -z "${OPENPI_PI0_PYTORCH_WEIGHT:-}" ]]; then
        echo "[err] OPENPI_PI0_JAX_WEIGHT and OPENPI_PI0_PYTORCH_WEIGHT must be set."; exit 1
    fi
    resume_flag=(); [[ "${RESUME}" == "1" ]] && resume_flag+=(--resume)
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
        "${lora_env[@]}" \
        uv --project "${OPENPI_ROOT}" run \
        "${launcher[@]}" python "${OPENPI_ROOT}/scripts/train_pytorch.py" \
        "${POLICY_CONFIG}" \
        --exp_name "${EXP_NAME}" \
        --num_train_steps "${TRAIN_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --save_interval "${SAVE_INTERVAL}" \
        --checkpoint_base_dir "${OUTPUT_DIR}" \
        "${resume_flag[@]}"
fi

# ---------------------------------------------------------------------------
# [6] eval (audio-mode slots_uv) on the three tasks
# ---------------------------------------------------------------------------
# Eval-time oracle noise. The openpi pi0 pipeline trains on the CLEAN stored
# audio columns (convert writes conf=1.0 and un-noised uv/energy/class), so eval
# defaults to clean too (all noise 0, conf 1.0) for train/eval parity. Override
# to probe robustness, or add matching train-time jitter for the noisy regime.
EVAL_NOISE_AZ_STD="${EVAL_NOISE_AZ_STD:-0}"
EVAL_NOISE_EL_STD="${EVAL_NOISE_EL_STD:-0}"
EVAL_NOISE_CONF_MIN="${EVAL_NOISE_CONF_MIN:-1.0}"
EVAL_NOISE_CONF_MAX="${EVAL_NOISE_CONF_MAX:-1.0}"
EVAL_NOISE_CLASS_FLIP_PROB="${EVAL_NOISE_CLASS_FLIP_PROB:-0}"
EVAL_NOISE_ENERGY_STD="${EVAL_NOISE_ENERGY_STD:-0}"

run_eval() {
    local task="$1" outdir="$2"
    echo "==================== [6] eval ${task} ===================="
    EVAL_DIR="${outdir}" TASK_NAME="${task}" TAXONOMY="${TAXONOMY}" \
    LOCAL=1 POLICY_CONFIG="${POLICY_CONFIG}" POLICY_DIR="${POLICY_DIR}" \
    OPENPI_ROOT="${OPENPI_ROOT}" AUDIO_MODE="slots_uv" EVAL_MODE="oracle" \
    EVAL_N="${EVAL_N}" EVAL_MAX_LEN="${EVAL_MAX_LEN}" EVAL_HORIZON="${EVAL_HORIZON}" \
    TOP_K="${TOP_K}" \
    NOISE_AZ_STD="${EVAL_NOISE_AZ_STD}" NOISE_EL_STD="${EVAL_NOISE_EL_STD}" \
    NOISE_CONF_MIN="${EVAL_NOISE_CONF_MIN}" NOISE_CONF_MAX="${EVAL_NOISE_CONF_MAX}" \
    NOISE_CLASS_FLIP_PROB="${EVAL_NOISE_CLASS_FLIP_PROB}" \
    NOISE_ENERGY_STD="${EVAL_NOISE_ENERGY_STD}" \
    bash sh/eval_pi05_audio.sh
}

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
    case "${EVAL_TASKS}" in
        all)
            run_eval select_radio        "${REPO_ROOT}/outputs/eval_seld_uv_select_radio"
            run_eval select_radio_two    "${REPO_ROOT}/outputs/eval_seld_uv_select_radio_two"
            run_eval select_radio_silent "${REPO_ROOT}/outputs/eval_seld_uv_select_radio_silent"
            ;;
        one)    run_eval select_radio        "${REPO_ROOT}/outputs/eval_seld_uv_select_radio" ;;
        two)    run_eval select_radio_two    "${REPO_ROOT}/outputs/eval_seld_uv_select_radio_two" ;;
        silent) run_eval select_radio_silent "${REPO_ROOT}/outputs/eval_seld_uv_select_radio_silent" ;;
        *) echo "[err] EVAL_TASKS must be all|one|two|silent"; exit 1 ;;
    esac
fi

echo "==================== [done] SELD-VLA pi0 slots_uv pipeline ===================="
echo "  lerobot     : ${LEROBOT_DIR}"
echo "  checkpoints : ${OUTPUT_DIR}/${EXP_NAME}/checkpoints"

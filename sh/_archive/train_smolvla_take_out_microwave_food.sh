#!/usr/bin/env bash
# End-to-end audio-conditioned SmolVLA pipeline for `take_out_microwave_food`.
#
# Unlike the radio tasks (continuous spatialised sources, classification +
# DOA), this task is *timing-based*: a one-shot microwave chime fires at a
# random time within the episode and the policy must wait for it, open the
# door within REACT_WINDOW_SEC seconds, then transfer the cooked food onto
# the tray. The expert sequence and failure detection both live in
# VLABench/tasks/hierarchical_tasks/composite/heat_food_series.py
# (`TakeOutMicrowaveFoodTask`).
#
# Because the cue is temporal — silence → chime → silence — the oracle path
# was extended to support *delayed* sources: extract_episode_gt records the
# microwave with `active_from_frame=trigger_step`, and the converter fills
# pre-trigger frames with silence in that slot. The oracle pipeline is the
# default here (USE_ORACLE=1) — it skips real audio synthesis (~10× faster
# at Stage 1) and trains the policy against the silence→active transition
# directly. Flip USE_ORACLE=0 to fall back to the binaural + real-SLED path.
#
# IMPORTANT — Stage 4 (eval): src/eval/eval_smolvla_audio.py is currently
# hard-coded to radio tasks (it pulls `active_radio_idx` from the config
# manager and bails if absent). Running it against take_out_microwave_food
# will crash on the first episode. Stage 4 is therefore disabled by default
# (DO_EVAL=0); the wiring remains so it "just works" once the eval script
# grows a take_out_microwave_food branch. Until then, use Stage 1's expert
# rollouts (the saved videos under ${DATASET_DIR}) as the success signal.
#
# Stage toggles:  DO_GENERATE / DO_CONVERT / DO_TRAIN / DO_EVAL  (0/1)
# Mode toggle:    USE_ORACLE                                    (0/1)
#
# Quick presets:
#
#   # smoke (≈ 1 min once ckpt is cached)
#   BATCH_SIZE=2 TRAIN_STEPS=5 WARMUP_STEPS=1 N_SAMPLE=10 \
#       DO_EVAL=0 bash sh/train_smolvla_take_out_microwave_food.sh
#
#   # sanity ("does the timing cue actually train?", ≈ 1h on RTX 5090)
#   N_SAMPLE=200 TRAIN_STEPS=4000 BATCH_SIZE=32 NUM_WORKERS=8 \
#       bash sh/train_smolvla_take_out_microwave_food.sh
#
#   # full (≈ 5h on RTX 5090)
#   N_SAMPLE=500 TRAIN_STEPS=15000 BATCH_SIZE=32 NUM_WORKERS=8 \
#       bash sh/train_smolvla_take_out_microwave_food.sh
#
#   # narrow the chime window to 5–10 s and tighten the reaction window
#   MIN_DELAY_SEC=5 MAX_DELAY_SEC=10 REACT_WINDOW_SEC=4 \
#       bash sh/train_smolvla_take_out_microwave_food.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT="/home/rllab/Desktop/AVLABench"
# Use distinct paths for oracle vs real-audio runs so they don't clobber
# each other when both modes are tried during development.v
_mode_tag="${USE_ORACLE:-1}"
if [[ "${_mode_tag}" == "1" ]]; then _mode_tag="oracle"; else _mode_tag="realaudio"; fi
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_take_out_microwave_food_${_mode_tag}}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_take_out_microwave_food_${_mode_tag}_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_take_out_microwave_food_${_mode_tag}}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_smolvla_take_out_microwave_food_${_mode_tag}}"

AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"

CONDA_ENV="${CONDA_ENV:-vlabench}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
REPO_ID="${REPO_ID:-local/avla_take_out_microwave_food}"
TASK_NAME="${TASK_NAME:-take_out_microwave_food}"

# ---------------------------------------------------------------------------
# Stage toggles
# ---------------------------------------------------------------------------
DO_GENERATE="${DO_GENERATE:-0}"
DO_CONVERT="${DO_CONVERT:-0}"
DO_TRAIN="${DO_TRAIN:-0}"
# DO_EVAL defaults to 0 — see header note re: radio-task hard-coding in
# src/eval/eval_smolvla_audio.py. Set DO_EVAL=1 once the eval script grows a
# take_out_microwave_food branch.
DO_EVAL="${DO_EVAL:-1}"

# Oracle (default) vs real-audio + SLED. Stage 1 records GT instead of
# synthesising audio, Stage 2 converts with --oracle-mode, Stage 3 reads
# the oracle_mode.json marker and re-samples fresh noise per batch.
USE_ORACLE="${USE_ORACLE:-1}"

# ---------------------------------------------------------------------------
# Task-specific timing knobs (passed to TakeOutMicrowaveFoodConfigManager via
# env vars). All values in *seconds*.
#   MIN/MAX_DELAY_SEC  → uniform sampling range for the chime trigger time
#   REACT_WINDOW_SEC   → grace period after the chime to open the microwave
# These must match between Stage 1 (generation) and Stage 4 (eval) — the eval
# env re-runs the task with the same code path, so simply leaving these env
# vars set across stages keeps the train/eval distributions aligned.
# ---------------------------------------------------------------------------
MIN_DELAY_SEC="${MIN_DELAY_SEC:-0.0}"
MAX_DELAY_SEC="${MAX_DELAY_SEC:-20.0}"
# 15 s is the smallest window where the oracle expert reliably opens the
# microwave in time — SkillLib.open_door interpolates the door arc at
# `target_velocity=0.05`, which costs ~15 s of sim time for this model.
# Drop this lower (e.g. 5) only for stricter evaluation regimes; data
# generation with REACT_WINDOW_SEC<10 will mostly fail with
# `door_not_opened_in_time` even when the expert grasp+open is otherwise
# correct.
REACT_WINDOW_SEC="${REACT_WINDOW_SEC:-15.0}"
export VLABENCH_MICROWAVE_MIN_DELAY_SEC="${MIN_DELAY_SEC}"
export VLABENCH_MICROWAVE_MAX_DELAY_SEC="${MAX_DELAY_SEC}"
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC="${REACT_WINDOW_SEC}"

# ---------------------------------------------------------------------------
# Stage 1 knobs — trajectory generation
# ---------------------------------------------------------------------------
N_SAMPLE="${N_SAMPLE:-2000}"
# Cap on the number of *saved* hdf5 episodes in the output dir. The
# generator's --max-episode arg defaults to 100, which silently truncates
# datasets that would otherwise grow past 100 even when N_SAMPLE is large.
# Keep this in sync with (or above) the headcount you actually want.
MAX_EPISODE="${MAX_EPISODE:-${N_SAMPLE}}"
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
# SLED quality gate is meant for continuous sources. For the one-shot chime
# only ~1s of audio is detectable per episode, so the default 0.4 confident-
# frame ratio would discard nearly every episode. Drop the gate by default;
# bump it back up if you want a stricter filter.
SLED_MIN_CONFIDENT_RATE="${SLED_MIN_CONFIDENT_RATE:-0.0}"
SLED_CONF_THRESH="${SLED_CONF_THRESH:-0.30}"

# ---------------------------------------------------------------------------
# Stage 2 knobs — HDF5 → LeRobot
# ---------------------------------------------------------------------------
TOP_K="${TOP_K:-3}"
AUDIO_MAX_LEN="${AUDIO_MAX_LEN:-96}"
VCODEC="${VCODEC:-h264}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# ---------------------------------------------------------------------------
# Stage 3 knobs — training
# ---------------------------------------------------------------------------
TRAIN_STEPS="${TRAIN_STEPS:-80000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
AUDIO_LR="${AUDIO_LR:-1e-4}"
LM_LR="${LM_LR:-}"
UNFREEZE_LM_LAYERS="${UNFREEZE_LM_LAYERS:-2}"
AUDIO_FUSION_MODE="${AUDIO_FUSION_MODE:-natural_language}"
CLASS_TOKEN_SCALE="${CLASS_TOKEN_SCALE:-0.1}"

# Train-time augmentation / regularisation. ZERO defaults make the policy
# overfit to the exact pixel & state distribution of the expert demos —
# this is the dominant failure mode on long-horizon manipulation tasks
# (the gripper visibly "jitters" near the target because the policy is
# brittle to even mm-level state drift it never saw during training).
# Recommended on for take_out_microwave_food.
STATE_NOISE_STD="${STATE_NOISE_STD:-0.005}"
IMAGE_COLOR_JITTER="${IMAGE_COLOR_JITTER:-0.1}"
IMAGE_TRANSLATE_PX="${IMAGE_TRANSLATE_PX:-2}"
AUDIO_CONF_DROPOUT="${AUDIO_CONF_DROPOUT:-0.05}"
DIRECTION_DROPOUT="${DIRECTION_DROPOUT:-0.05}"

# VLM LoRA (full-layer text-model fine-tuning via low-rank adapters).
# When USE_VLM_LORA=1 the trainer ignores UNFREEZE_LM_LAYERS and instead
# injects LoRA adapters into every text-model transformer block. Adapters
# are merged into the base weights at save time so the saved ckpt is
# loadable by the standard (non-LoRA) eval script. Default targets are
# the full attention + MLP linear set (q,k,v,o,gate,up,down) — i.e. true
# "full-layer LoRA". Switch to "q_proj,v_proj" for the lighter variant.
USE_VLM_LORA="${USE_VLM_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Oracle noise knobs (only used when USE_ORACLE=1; see train_smolvla_oracle.sh
# for the rationale behind these defaults).
NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"

# Resume from a previous ckpt (empty = train from scratch)
RESUME_FROM="${RESUME_FROM:-}"

_warmup_default=$(( TRAIN_STEPS / 20 ))
if   (( _warmup_default > 500 )); then _warmup_default=500
elif (( _warmup_default < 1   )); then _warmup_default=1
fi
WARMUP_STEPS="${WARMUP_STEPS:-${_warmup_default}}"

_save_default=$(( TRAIN_STEPS / 10 ))
if (( _save_default < 1 )); then _save_default="${TRAIN_STEPS}"; fi
SAVE_EVERY="${SAVE_EVERY:-${_save_default}}"
LOG_EVERY="${LOG_EVERY:-20}"

# ---------------------------------------------------------------------------
# Stage 4 knobs — eval (real audio + SLED)
# ---------------------------------------------------------------------------
EVAL_N="${EVAL_N:-100}"
# Episode budget needs to cover the worst-case wait + the manipulation chain.
# Rough rule of thumb: ceil((MAX_DELAY_SEC + REACT_WINDOW_SEC) * fps) + ~150
# steps for open_door → pick → place. Override with EVAL_MAX_LEN=N if needed.
_eval_max_default=$(python3 -c "import math, os; print(int(math.ceil((${MAX_DELAY_SEC} + ${REACT_WINDOW_SEC}) * 10) + 200))")
EVAL_MAX_LEN="${EVAL_MAX_LEN:-${_eval_max_default}}"
# Larger horizon = more actions executed per re-inference, which gives
# *much* smoother motion on long-horizon tasks. The chunk boundary at
# horizon=5 was causing visible jitter every 0.5 s. SmolVLA's predicted
# action chunk is typically 50, so 10–15 still re-plans well before the
# chunk runs out.
EVAL_HORIZON="${EVAL_HORIZON:-10}"
EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"
EVAL_INTENTION_MODE="${EVAL_INTENTION_MODE:-exclusive}"
EVAL_EXCLUSIVE_INTENTION_MARGIN="${EVAL_EXCLUSIVE_INTENTION_MARGIN:-0.03}"
EVAL_WARMUP_SECONDS="${EVAL_WARMUP_SECONDS:-3.0}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_SAVE_AUDIO="${EVAL_SAVE_AUDIO:-0}"
EVAL_SAVE_AUDIO_LOG="${EVAL_SAVE_AUDIO_LOG:-0}"

# ---------------------------------------------------------------------------
PY="conda run -n ${CONDA_ENV} --no-capture-output python"
cd "${REPO_ROOT}"

cat <<EOF
[paths]    repo      = ${REPO_ROOT}
           dataset   = ${DATASET_DIR}
           lerobot   = ${LEROBOT_DIR}
           output    = ${OUTPUT_DIR}
           eval_dir  = ${EVAL_DIR}
[mode]     USE_ORACLE=${USE_ORACLE}  (1 = GT geometry + noise; 0 = real binaural + SLED)
[task]     ${TASK_NAME}
           chime delay   ~ U[${MIN_DELAY_SEC}, ${MAX_DELAY_SEC}] s
           react window  = ${REACT_WINDOW_SEC} s
[stages]   gen=${DO_GENERATE} convert=${DO_CONVERT} train=${DO_TRAIN} eval=${DO_EVAL}
[stage 1]  n_sample=${N_SAMPLE}  start_idle=${START_IDLE_SECONDS}s
           sled_gate: min_rate=${SLED_MIN_CONFIDENT_RATE} thresh=${SLED_CONF_THRESH}
[stage 2]  top_k=${TOP_K}  image=${IMAGE_SIZE}  vcodec=${VCODEC}
[stage 3]  steps=${TRAIN_STEPS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}
           lr=${LR} audio_lr=${AUDIO_LR} lm_lr=${LM_LR:-auto}
           unfreeze_lm=${UNFREEZE_LM_LAYERS}  warmup=${WARMUP_STEPS}
           audio_fusion=${AUDIO_FUSION_MODE} class_token_scale=${CLASS_TOKEN_SCALE}
           vlm_lora=${USE_VLM_LORA} r=${LORA_R} alpha=${LORA_ALPHA} targets=${LORA_TARGETS}
           aug: state_noise=${STATE_NOISE_STD} color_jitter=${IMAGE_COLOR_JITTER}
                translate_px=${IMAGE_TRANSLATE_PX} audio_conf_drop=${AUDIO_CONF_DROPOUT}
                direction_drop=${DIRECTION_DROPOUT}
[stage 4]  n=${EVAL_N}  max_len=${EVAL_MAX_LEN}  horizon=${EVAL_HORIZON}
           intention_threshold=${EVAL_INTENTION_THRESHOLD}
           intention_mode=${EVAL_INTENTION_MODE} exclusive_margin=${EVAL_EXCLUSIVE_INTENTION_MARGIN}
           warmup=${EVAL_WARMUP_SECONDS}s
           eval_dir=${EVAL_DIR}
EOF

# ---------------------------------------------------------------------------
# Stage 1 — trajectory generation
#           ORACLE: skip audio synthesis + SLED; record GT geometry with
#                   active_from_frame=trigger_step into HDF5.
#           REAL  : synthesise binaural audio (chime gated by start_step)
#                   and run SLED to produce per-frame predictions.
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    echo "============================================================"
    if [[ "${USE_ORACLE}" == "1" ]]; then
        echo "[1/4] generating up to ${N_SAMPLE} ORACLE trajectories (cap=${MAX_EPISODE})"
        echo "============================================================"
        ${PY} scripts/trajectory_generation.py \
            --task-name    "${TASK_NAME}" \
            --oracle-mode \
            --save-dir     "${DATASET_DIR}" \
            --n-sample     "${N_SAMPLE}" \
            --max-episode  "${MAX_EPISODE}" \
            --start-idle-seconds "${START_IDLE_SECONDS}"
    else
        echo "[1/4] generating up to ${N_SAMPLE} REAL-AUDIO trajectories (cap=${MAX_EPISODE})"
        echo "============================================================"
        ${PY} scripts/trajectory_generation.py \
            --task-name    "${TASK_NAME}" \
            --audio-config "${AUDIO_CONFIG}" \
            --sled-ckpt    "${SLED_CKPT}" \
            --save-dir     "${DATASET_DIR}" \
            --n-sample     "${N_SAMPLE}" \
            --max-episode  "${MAX_EPISODE}" \
            --start-idle-seconds "${START_IDLE_SECONDS}" \
            --sled-min-confident-rate "${SLED_MIN_CONFIDENT_RATE}" \
            --sled-conf-thresh        "${SLED_CONF_THRESH}"
    fi

    n_h5=$(ls "${DATASET_DIR}/${TASK_NAME}"/data_*.hdf5 2>/dev/null | wc -l)
    echo "[1/4] HDF5 episodes: ${n_h5}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — HDF5 → LeRobot. Oracle path reads meta_info/oracle_audio; real
# path reads meta_info/sled_predictions.
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "============================================================"
    echo "[2/4] converting → ${LEROBOT_DIR} (oracle=${USE_ORACLE})"
    echo "============================================================"
    rm -rf "${LEROBOT_DIR}"
    convert_args=()
    if [[ "${USE_ORACLE}" == "1" ]]; then
        convert_args+=(--oracle-mode)
    fi
    # take_out_microwave_food stores a task-specific English instruction
    # ("When the microwave chimes, open the door, …") inside each HDF5;
    # we must propagate it to LeRobot instead of letting the converter
    # bake in its default select_radio prompt. Eval-time
    # `env.task.get_instruction()` returns the same shape, so the policy
    # sees a matching prompt during training and inference.
    convert_args+=(--use-episode-instruction)
    ${PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${DATASET_DIR}/${TASK_NAME}" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps     10 \
        --top-k   "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec  "${VCODEC}" \
        "${convert_args[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 3 — fine-tune the audio-aware SmolVLA. Trainer auto-detects oracle
# datasets via the oracle_mode.json marker dropped by the converter, and
# resamples fresh noise per batch. When USE_ORACLE=1 we also pass the
# noise knobs explicitly so they're easy to tweak from the env.
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "============================================================"
    echo "[3/4] training (${TRAIN_STEPS} steps, batch=${BATCH_SIZE}, oracle=${USE_ORACLE})"
    echo "============================================================"
    if [[ -z "${RESUME_FROM}" ]]; then
        rm -rf "${OUTPUT_DIR}"
    else
        echo "[resume] keeping ${OUTPUT_DIR} (resuming from ${RESUME_FROM})"
    fi
    lm_lr_args=()
    if [[ -n "${LM_LR}" ]]; then
        lm_lr_args+=(--lm-lr "${LM_LR}")
    fi
    resume_args=()
    if [[ -n "${RESUME_FROM}" ]]; then
        resume_args+=(--resume-from "${RESUME_FROM}")
    fi
    oracle_noise_args=()
    if [[ "${USE_ORACLE}" == "1" ]]; then
        oracle_noise_args+=(
            --oracle-noise on
            --noise-az-std        "${NOISE_AZ_STD}"
            --noise-el-std        "${NOISE_EL_STD}"
            --noise-conf-min      "${NOISE_CONF_MIN}"
            --noise-conf-max      "${NOISE_CONF_MAX}"
            --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}"
        )
    fi
    lora_args=()
    if [[ "${USE_VLM_LORA}" == "1" ]]; then
        # When LoRA is on the trainer auto-zeroes unfreeze_last_n_lm_layers
        # (no conflict from leaving UNFREEZE_LM_LAYERS set above), and
        # defaults lm_lr to `lr` instead of `lr/5`.
        lora_args+=(
            --vlm-lora
            --lora-r              "${LORA_R}"
            --lora-alpha          "${LORA_ALPHA}"
            --lora-dropout        "${LORA_DROPOUT}"
            --lora-target-modules "${LORA_TARGETS}"
        )
        echo "[lora] enabled: r=${LORA_R} alpha=${LORA_ALPHA} targets=${LORA_TARGETS}"
    fi
    ${PY} src/training/train_smolvla_audio.py \
        --dataset-root "${LEROBOT_DIR}" \
        --repo-id      "${REPO_ID}" \
        --pretrained   "${PRETRAINED}" \
        --taxonomy     "${TAXONOMY}" \
        --output-dir   "${OUTPUT_DIR}" \
        --audio-max-len "${AUDIO_MAX_LEN}" \
        --batch-size   "${BATCH_SIZE}" \
        --steps        "${TRAIN_STEPS}" \
        --lr           "${LR}" \
        --audio-lr     "${AUDIO_LR}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --num-workers  "${NUM_WORKERS}" \
        --save-every   "${SAVE_EVERY}" \
        --log-every    "${LOG_EVERY}" \
        --unfreeze-last-n-lm-layers "${UNFREEZE_LM_LAYERS}" \
        --audio-fusion-mode "${AUDIO_FUSION_MODE}" \
        --class-token-scale "${CLASS_TOKEN_SCALE}" \
        --state-noise-std    "${STATE_NOISE_STD}" \
        --image-color-jitter "${IMAGE_COLOR_JITTER}" \
        --image-translate-px "${IMAGE_TRANSLATE_PX}" \
        --audio-conf-dropout "${AUDIO_CONF_DROPOUT}" \
        --direction-dropout  "${DIRECTION_DROPOUT}" \
        "${oracle_noise_args[@]}" \
        "${lora_args[@]}" \
        "${lm_lr_args[@]}" \
        "${resume_args[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 4 — eval with real audio + SLED
# (The eval env reads the same VLABENCH_MICROWAVE_* env vars set above, so
#  train- and eval-time chime distributions stay aligned by default.)
# ---------------------------------------------------------------------------
if [[ "${DO_EVAL}" == "1" ]]; then
    CKPT="$(ls -t ${OUTPUT_DIR}/ckpt_step*.pt 2>/dev/null | head -1)"
    if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
        echo "[err] no ckpt found in ${OUTPUT_DIR}"
        exit 1
    fi
    if [[ ! -f "${SLED_CKPT}" ]]; then
        echo "[err] SLED_CKPT not found: ${SLED_CKPT}"
        exit 1
    fi
    if [[ ! -f "${AUDIO_CONFIG}" ]]; then
        echo "[err] AUDIO_CONFIG not found: ${AUDIO_CONFIG}"
        exit 1
    fi

    extra=()
    [[ "${EVAL_SAVE_VIDEO}"     == "1" ]] && extra+=(--save-video)
    [[ "${EVAL_SAVE_AUDIO}"     == "1" ]] && extra+=(--save-audio)
    [[ "${EVAL_SAVE_AUDIO_LOG}" == "1" ]] && extra+=(--save-audio-log)

    echo "============================================================"
    echo "[4/4] eval REAL SLED  ${EVAL_N} episodes"
    echo "       ckpt    =${CKPT}"
    echo "       audio   =${AUDIO_CONFIG}"
    echo "       sled    =${SLED_CKPT}"
    echo "       warm-up =${EVAL_WARMUP_SECONDS}s"
    echo "       out     =${EVAL_DIR}"
    echo "============================================================"
    ${PY} src/eval/eval_smolvla_audio.py \
        --ckpt        "${CKPT}" \
        --pretrained  "${PRETRAINED}" \
        --taxonomy    "${TAXONOMY}" \
        --task-name   "${TASK_NAME}" \
        --n-episodes  "${EVAL_N}" \
        --max-episode-length "${EVAL_MAX_LEN}" \
        --horizon     "${EVAL_HORIZON}" \
        --intention-threshold "${EVAL_INTENTION_THRESHOLD}" \
        --intention-mode "${EVAL_INTENTION_MODE}" \
        --exclusive-intention-margin "${EVAL_EXCLUSIVE_INTENTION_MARGIN}" \
        --warmup-seconds "${EVAL_WARMUP_SECONDS}" \
        --save-dir    "${EVAL_DIR}" \
        --audio-config "${AUDIO_CONFIG}" \
        --sled-ckpt    "${SLED_CKPT}" \
        "${extra[@]}"
fi

echo "============================================================"
echo "[done] take_out_microwave_food pipeline finished"
echo "  trajectories : ${DATASET_DIR}/${TASK_NAME}"
echo "  LeRobot      : ${LEROBOT_DIR}"
echo "  checkpoints  : ${OUTPUT_DIR}"
echo "  eval         : ${EVAL_DIR}"
echo "============================================================"

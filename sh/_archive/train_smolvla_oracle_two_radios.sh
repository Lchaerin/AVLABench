#!/usr/bin/env bash
# Oracle-mode end-to-end pipeline for the **two-radio** SmolVLA task.
#
# Variant of sh/train_smolvla_oracle.sh:
#   * Stage 1 generates trajectories for `select_radio_two`. Two of the three
#     radios are randomly active each episode and emit sounds from two
#     *different* taxonomy classes; the per-episode instruction names the
#     target sound class. The robot must press the button in front of the
#     radio playing that class.
#   * Stage 2 converts HDF5 → LeRobot with --use-episode-instruction so the
#     class-specific prompt saved per episode is preserved (vs. the static
#     prompt the single-radio pipeline uses). Both active sources (target +
#     distractor) are stored so the network sees the full audio scene.
#   * Stage 3 trains as in the single-radio oracle script (auto-detects the
#     oracle dataset; re-samples noise per batch on top of GT).
#     KEY DIFFERENCE from the single-radio script: both audio slots are active.
#     The network must internally match the instruction's class name against
#     the audio source descriptions to identify which direction belongs to the
#     target radio. shuffle_slots=on breaks the slot-0=target positional
#     shortcut; target_slot_protect=on keeps the target's class_id stable
#     during noise injection.
#   * Stage 4 evaluates with --oracle-mode by default, picking 2 sounds per
#     episode and constructing the matching instruction at runtime.
#
# Stage toggles:  DO_GENERATE / DO_CONVERT / DO_TRAIN / DO_EVAL  (0/1)

set -euo pipefail

# ---------------------------------------------------------------------------
REPO_ROOT="/home/rllab/Desktop/AVLABench"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/dataset_oracle_two}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_oracle_two_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/smolvla_oracle_two}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_smolvla_oracle_two}"

TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"

CONDA_ENV="${CONDA_ENV:-vlabench}"
PRETRAINED="${PRETRAINED:-lerobot/smolvla_vlabench}"
REPO_ID="${REPO_ID:-local/avla_select_radio_two_oracle}"

# ---------------------------------------------------------------------------
# Stage toggles
# ---------------------------------------------------------------------------
DO_GENERATE="${DO_GENERATE:-0}"
DO_CONVERT="${DO_CONVERT:-1}"   # re-convert: both sources stored (target + distractor)
DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"

# ---------------------------------------------------------------------------
# Stage 1 knobs — trajectory generation (oracle mode, no audio)
# ---------------------------------------------------------------------------
N_SAMPLE="${N_SAMPLE:-500}"
TASK_NAME="${TASK_NAME:-select_radio_two}"
# Cap on how many HDF5 episodes can sit in the task dir before the
# generator early-exits. Defaults to N_SAMPLE so this knob does not silently
# clamp Stage 1 below what the user asked for (the script's own default in
# trajectory_generation.py is 100).
MAX_EPISODE="${MAX_EPISODE:-${N_SAMPLE}}"

# Hold the robot still at episode start so the audio pipeline (and the
# trained policy) get a brief window before the expert begins to act.
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"

# ---------------------------------------------------------------------------
# Stage 2 knobs — HDF5 → LeRobot
# ---------------------------------------------------------------------------
TOP_K="${TOP_K:-3}"
AUDIO_MAX_LEN="${AUDIO_MAX_LEN:-96}"
VCODEC="${VCODEC:-h264}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

# ---------------------------------------------------------------------------
# Stage 3 knobs — training
#
# Both audio slots (target + distractor) are present in each training sample.
# shuffle_slots=on breaks the positional shortcut (slot 0 = target) so the
# network must use class-name matching from the instruction to identify the
# correct direction token. target_slot_protect=on prevents the target's
# class_id from being flipped during noise injection (slot 0 before the
# shuffle, i.e. the target), keeping the instruction↔class supervision stable.
# ---------------------------------------------------------------------------
TRAIN_STEPS="${TRAIN_STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-5e-5}"
AUDIO_LR="${AUDIO_LR:-1e-4}"
LM_LR="${LM_LR:-}"                          # empty → trainer uses lr/5
UNFREEZE_LM_LAYERS="${UNFREEZE_LM_LAYERS:-2}"
AUDIO_FUSION_MODE="${AUDIO_FUSION_MODE:-natural_language}"
CLASS_TOKEN_SCALE="${CLASS_TOKEN_SCALE:-0.1}"

# Multi-source augmentations — required for two-radio task.
# shuffle_slots: break the slot-0=target positional shortcut.
# target_slot_protect: keep the target class stable across flip augmentation.
# target_first_slots: target-aware canonicalization ablation. Keep off for the
# standard benchmark; it can make the policy rely on target-conditioned slot
# order instead of VLM class-name matching.
SHUFFLE_SLOTS="${SHUFFLE_SLOTS:-on}"
TARGET_SLOT_PROTECT="${TARGET_SLOT_PROTECT:-on}"
TARGET_FIRST_SLOTS="${TARGET_FIRST_SLOTS:-off}"
CANONICALIZE_SLOTS="${CANONICALIZE_SLOTS:-azimuth}"

# Oracle noise knobs (tune → how close to "perfect SLED" we train against)
NOISE_AZ_STD="${NOISE_AZ_STD:-3.0}"
NOISE_EL_STD="${NOISE_EL_STD:-5.0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-0.85}"
NOISE_CONF_MAX="${NOISE_CONF_MAX:-0.98}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0.02}"

# Train-time regularisation knobs. These are shared with the single-radio
# script; defaults are conservative and can be overridden per variant.
IMAGE_COLOR_JITTER="${IMAGE_COLOR_JITTER:-0.0}"
IMAGE_TRANSLATE_PX="${IMAGE_TRANSLATE_PX:-0}"
STATE_NOISE_STD="${STATE_NOISE_STD:-0.0}"
AUDIO_CONF_DROPOUT="${AUDIO_CONF_DROPOUT:-0.0}"
DIRECTION_DROPOUT="${DIRECTION_DROPOUT:-0.0}"
DIRECTION_ENCODER_TYPE="${DIRECTION_ENCODER_TYPE:-mlp}"

# Optional LoRA on the VLM language tower. Disabled by default because the
# full target set can OOM at batch=32; enable per variant with VLM_LORA=1.
VLM_LORA="${VLM_LORA:-0}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,v_proj}"

# Resume from a previous ckpt (empty = train from scratch).
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
# Stage 4 knobs — eval
# ---------------------------------------------------------------------------
# EVAL_MODE — same semantics as the single-radio script:
#   oracle    — bypass SLED, build top-K from env GT + noise.
#   real_sled — synthesise binaural audio from BOTH active radios + run SLED.
#   both      — run both back-to-back into separate output dirs.
EVAL_MODE="${EVAL_MODE:-oracle}"

EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-200}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
EVAL_INTENTION_THRESHOLD="${EVAL_INTENTION_THRESHOLD:-0.1}"
EVAL_INTENTION_MODE="${EVAL_INTENTION_MODE:-exclusive}"
EVAL_EXCLUSIVE_INTENTION_MARGIN="${EVAL_EXCLUSIVE_INTENTION_MARGIN:-0.03}"
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}"
EVAL_NO_INSTRUCTION="${EVAL_NO_INSTRUCTION:-0}"

# Non-ML post-processing: temporal majority vote on the top-K class IDs
# the policy receives (suppresses single-frame class flips coming from
# the noisy oracle/SLED snapshot). Set to 0 to disable.
CLASS_SMOOTHING_WINDOW="${CLASS_SMOOTHING_WINDOW:-5}"
SHUFFLE_AUDIO_SLOTS_EVAL="${SHUFFLE_AUDIO_SLOTS_EVAL:-on}"
TARGET_FIRST_AUDIO_SLOTS_EVAL="${TARGET_FIRST_AUDIO_SLOTS_EVAL:-off}"
CANONICALIZE_AUDIO_SLOTS_EVAL="${CANONICALIZE_AUDIO_SLOTS_EVAL:-azimuth}"

# Real-SLED-only knobs (ignored when EVAL_MODE=oracle)
AUDIO_CONFIG="${AUDIO_CONFIG:-${REPO_ROOT}/audio_generation/scene_audio_config.json}"
SLED_CKPT="${SLED_CKPT:-/home/rllab/Desktop/crossCorr/sled_v5/checkpoints_ver11/biseld_best.pt}"
EVAL_WARMUP_SECONDS="${EVAL_WARMUP_SECONDS:-3.0}"
EVAL_SAVE_AUDIO="${EVAL_SAVE_AUDIO:-0}"
EVAL_SAVE_AUDIO_LOG="${EVAL_SAVE_AUDIO_LOG:-0}"

EVAL_DIR_ORACLE="${EVAL_DIR_ORACLE:-${EVAL_DIR}}"
EVAL_DIR_REAL_SLED="${EVAL_DIR_REAL_SLED:-${REPO_ROOT}/outputs/eval_smolvla_oracle_two_realsled}"

# ---------------------------------------------------------------------------
PY="conda run -n ${CONDA_ENV} --no-capture-output python"
cd "${REPO_ROOT}"

cat <<EOF
[paths]    repo      = ${REPO_ROOT}
           dataset   = ${DATASET_DIR}
           lerobot   = ${LEROBOT_DIR}
           output    = ${OUTPUT_DIR}
           eval_dir  = ${EVAL_DIR}
[stages]   gen=${DO_GENERATE} convert=${DO_CONVERT} train=${DO_TRAIN} eval=${DO_EVAL}
[stage 1]  task=${TASK_NAME}  n_sample=${N_SAMPLE}   (oracle, two radios, no real audio)
           idle=${START_IDLE_SECONDS}s @ ${DATASET_FPS}fps
[stage 2]  top_k=${TOP_K}  image=${IMAGE_SIZE}  vcodec=${VCODEC}  per-episode instruction
[stage 3]  steps=${TRAIN_STEPS} batch=${BATCH_SIZE} workers=${NUM_WORKERS}
           lr=${LR} audio_lr=${AUDIO_LR} lm_lr=${LM_LR:-auto}
           unfreeze_lm=${UNFREEZE_LM_LAYERS}  warmup=${WARMUP_STEPS}
           audio_fusion=${AUDIO_FUSION_MODE} class_token_scale=${CLASS_TOKEN_SCALE}
           noise: az±${NOISE_AZ_STD}° el±${NOISE_EL_STD}°
                  conf~U[${NOISE_CONF_MIN},${NOISE_CONF_MAX}]
                  flip=${NOISE_CLASS_FLIP_PROB}
           shuffle_slots=${SHUFFLE_SLOTS}  target_slot_protect=${TARGET_SLOT_PROTECT}
           canonicalize_slots=${CANONICALIZE_SLOTS}
           direction_encoder=${DIRECTION_ENCODER_TYPE}
[stage 4]  mode=${EVAL_MODE}  n=${EVAL_N}  max_len=${EVAL_MAX_LEN}  horizon=${EVAL_HORIZON}
           intention_threshold=${EVAL_INTENTION_THRESHOLD}
           intention_mode=${EVAL_INTENTION_MODE} exclusive_margin=${EVAL_EXCLUSIVE_INTENTION_MARGIN}
           class_smoothing_window=${CLASS_SMOOTHING_WINDOW}
           shuffle_audio_slots_eval=${SHUFFLE_AUDIO_SLOTS_EVAL}
           target_first_audio_slots_eval=${TARGET_FIRST_AUDIO_SLOTS_EVAL}
           canonicalize_audio_slots_eval=${CANONICALIZE_AUDIO_SLOTS_EVAL}
           oracle_dir=${EVAL_DIR_ORACLE}
           realsled_dir=${EVAL_DIR_REAL_SLED}
EOF

# ---------------------------------------------------------------------------
# Stage 1 — oracle trajectory generation (no audio/SLED)
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    echo "============================================================"
    echo "[1/4] generating up to ${N_SAMPLE} oracle trajectories (two-radio)"
    echo "============================================================"
    ${PY} scripts/trajectory_generation.py \
        --task-name    "${TASK_NAME}" \
        --oracle-mode \
        --save-dir     "${DATASET_DIR}" \
        --n-sample     "${N_SAMPLE}" \
        --max-episode  "${MAX_EPISODE}" \
        --start-idle-seconds "${START_IDLE_SECONDS}" \
        --dataset-fps  "${DATASET_FPS}"

    n_h5=$(ls "${DATASET_DIR}/${TASK_NAME}"/data_*.hdf5 2>/dev/null | wc -l)
    echo "[1/4] HDF5 episodes: ${n_h5}"
fi

# ---------------------------------------------------------------------------
# Stage 2 — HDF5 → LeRobot (oracle, per-episode instruction)
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    echo "============================================================"
    echo "[2/4] converting → ${LEROBOT_DIR} (oracle mode, per-episode instruction)"
    echo "============================================================"
    rm -rf "${LEROBOT_DIR}"
    ${PY} src/data/convert_hdf5_to_lerobot.py \
        --src-dir "${DATASET_DIR}/${TASK_NAME}" \
        --out-dir "${LEROBOT_DIR}" \
        --repo-id "${REPO_ID}" \
        --fps     10 \
        --top-k   "${TOP_K}" \
        --image-h "${IMAGE_SIZE}" \
        --image-w "${IMAGE_SIZE}" \
        --vcodec  "${VCODEC}" \
        --oracle-mode \
        --use-episode-instruction
fi

# ---------------------------------------------------------------------------
# Stage 3 — train (auto-detects oracle via oracle_mode.json marker)
# ---------------------------------------------------------------------------
if [[ "${DO_TRAIN}" == "1" ]]; then
    echo "============================================================"
    echo "[3/4] training (${TRAIN_STEPS} steps, batch=${BATCH_SIZE})"
    echo "============================================================"
    if [[ -n "${RESUME_FROM}" ]]; then
        echo "[resume] keeping ${OUTPUT_DIR} (resuming from ${RESUME_FROM})"
        mkdir -p "${OUTPUT_DIR}"
    else
        rm -rf "${OUTPUT_DIR}"
    fi
    lm_lr_args=()
    if [[ -n "${LM_LR}" ]]; then
        lm_lr_args+=(--lm-lr "${LM_LR}")
    fi
    lora_args=()
    if [[ "${VLM_LORA}" == "1" ]]; then
        lora_args+=(
            --vlm-lora
            --lora-r "${LORA_R}"
            --lora-alpha "${LORA_ALPHA}"
            --lora-dropout "${LORA_DROPOUT}"
            --lora-target-modules "${LORA_TARGET_MODULES}"
        )
    fi
    resume_args=()
    if [[ -n "${RESUME_FROM}" ]]; then
        resume_args+=(--resume-from "${RESUME_FROM}")
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
        --oracle-noise on \
        --noise-az-std        "${NOISE_AZ_STD}" \
        --noise-el-std        "${NOISE_EL_STD}" \
        --noise-conf-min      "${NOISE_CONF_MIN}" \
        --noise-conf-max      "${NOISE_CONF_MAX}" \
        --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
        --shuffle-slots         "${SHUFFLE_SLOTS}" \
        --target-slot-protect   "${TARGET_SLOT_PROTECT}" \
        --target-first-slots    "${TARGET_FIRST_SLOTS}" \
        --canonicalize-slots    "${CANONICALIZE_SLOTS}" \
        --image-color-jitter "${IMAGE_COLOR_JITTER}" \
        --image-translate-px "${IMAGE_TRANSLATE_PX}" \
        --state-noise-std "${STATE_NOISE_STD}" \
        --audio-conf-dropout "${AUDIO_CONF_DROPOUT}" \
        --direction-dropout "${DIRECTION_DROPOUT}" \
        --direction-encoder-type "${DIRECTION_ENCODER_TYPE}" \
        "${lm_lr_args[@]}" \
        "${lora_args[@]}" \
        "${resume_args[@]}"
fi

# ---------------------------------------------------------------------------
# Stage 4 — eval. Same checkpoint, two possible audio paths:
#   * oracle    — env GT + noise (no SLED model in the loop)
#   * real_sled — synthesise binaural audio (both radios) + run SLED online
# ---------------------------------------------------------------------------
if [[ "${DO_EVAL}" == "1" ]]; then
    case "${EVAL_MODE}" in
        oracle|real_sled|both) ;;
        *) echo "[err] EVAL_MODE must be one of: oracle | real_sled | both"; exit 1 ;;
    esac
    CKPT="$(ls -t ${OUTPUT_DIR}/ckpt_step*.pt 2>/dev/null | head -1)"
    if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
        echo "[err] no ckpt found in ${OUTPUT_DIR}"
        exit 1
    fi
    extra=()
    [[ "${EVAL_SAVE_VIDEO}"     == "1" ]] && extra+=(--save-video)
    [[ "${EVAL_SAVE_AUDIO}"     == "1" ]] && extra+=(--save-audio)
    [[ "${EVAL_SAVE_AUDIO_LOG}" == "1" ]] && extra+=(--save-audio-log)
    [[ "${EVAL_NO_INSTRUCTION}" == "1" ]] && extra+=(--no-instruction)

    if [[ "${EVAL_MODE}" == "oracle" || "${EVAL_MODE}" == "both" ]]; then
        echo "============================================================"
        echo "[4/4] eval ORACLE   (no SLED model)  ${EVAL_N} episodes"
        echo "       ckpt=${CKPT}"
        echo "       out =${EVAL_DIR_ORACLE}"
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
            --save-dir    "${EVAL_DIR_ORACLE}" \
            --oracle-mode \
            --noise-az-std        "${NOISE_AZ_STD}" \
            --noise-el-std        "${NOISE_EL_STD}" \
            --noise-conf-min      "${NOISE_CONF_MIN}" \
            --noise-conf-max      "${NOISE_CONF_MAX}" \
            --noise-class-flip-prob "${NOISE_CLASS_FLIP_PROB}" \
            --class-smoothing-window "${CLASS_SMOOTHING_WINDOW}" \
            --shuffle-audio-slots "${SHUFFLE_AUDIO_SLOTS_EVAL}" \
            --target-first-audio-slots "${TARGET_FIRST_AUDIO_SLOTS_EVAL}" \
            --canonicalize-audio-slots "${CANONICALIZE_AUDIO_SLOTS_EVAL}" \
            "${extra[@]}"
    fi

    if [[ "${EVAL_MODE}" == "real_sled" || "${EVAL_MODE}" == "both" ]]; then
        if [[ ! -f "${SLED_CKPT}" ]]; then
            echo "[err] real_sled mode but SLED_CKPT not found: ${SLED_CKPT}"
            exit 1
        fi
        if [[ ! -f "${AUDIO_CONFIG}" ]]; then
            echo "[err] real_sled mode but AUDIO_CONFIG not found: ${AUDIO_CONFIG}"
            exit 1
        fi
        echo "============================================================"
        echo "[4/4] eval REAL SLED  (binaural audio + SLED online)  ${EVAL_N} episodes"
        echo "       ckpt    =${CKPT}"
        echo "       audio   =${AUDIO_CONFIG}"
        echo "       sled    =${SLED_CKPT}"
        echo "       warm-up =${EVAL_WARMUP_SECONDS}s"
        echo "       out     =${EVAL_DIR_REAL_SLED}"
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
            --save-dir    "${EVAL_DIR_REAL_SLED}" \
            --audio-config "${AUDIO_CONFIG}" \
            --sled-ckpt    "${SLED_CKPT}" \
            --class-smoothing-window "${CLASS_SMOOTHING_WINDOW}" \
            --shuffle-audio-slots "${SHUFFLE_AUDIO_SLOTS_EVAL}" \
            --target-first-audio-slots "${TARGET_FIRST_AUDIO_SLOTS_EVAL}" \
            --canonicalize-audio-slots "${CANONICALIZE_AUDIO_SLOTS_EVAL}" \
            "${extra[@]}"
    fi
fi

echo "============================================================"
echo "[done] oracle two-radio pipeline finished"
echo "  trajectories   : ${DATASET_DIR}/${TASK_NAME}"
echo "  LeRobot        : ${LEROBOT_DIR}"
echo "  checkpoints    : ${OUTPUT_DIR}"
case "${EVAL_MODE}" in
    oracle)    echo "  eval (oracle)  : ${EVAL_DIR_ORACLE}" ;;
    real_sled) echo "  eval (sled)    : ${EVAL_DIR_REAL_SLED}" ;;
    both)      echo "  eval (oracle)  : ${EVAL_DIR_ORACLE}"
               echo "  eval (sled)    : ${EVAL_DIR_REAL_SLED}" ;;
esac
echo "============================================================"

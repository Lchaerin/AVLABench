#!/usr/bin/env bash
# Evaluate the pi0.5 find_hidden_object_open policy.
#
# Default: all four slots ({left,right} x {top,bottom}) — matching the training
# set. The eval already reports a per-slot breakdown, so an all-slots run is
# what measures the **elevation** skill (top vs bottom) as opposed to azimuth
# (left vs right).
#
# SLOTS restricts the draw when you want a focused run; VLABENCH_HIDDEN_SLOT_LABEL
# now takes a comma-separated set and picks uniformly per episode:
#   SLOTS=left_top,right_top   ...   top drawers only
#   SLOTS=left_bottom,right_bottom   bottom drawers only (elevation stress test)
#   SLOTS=                     ...   all four (default)
#
# MIC_CAM picks the camera the audio (u,v) is projected through. Left unset it is
# inferred from the checkpoint's own training dataset (see below), which is what
# you want — pass it only to deliberately mismatch train/eval for a diagnostic.
#
# Unlike sh/train_pi0_find_hidden.sh (whose EVAL_DIR is hardcoded and overwrites
# the previous run), EVAL_DIR here defaults to a per-checkpoint directory.
#
#   POLICY_DIR=outputs/pi05_find_hidden_stagedfreeze/<config>/<exp>/10000 \
#     bash sh/eval_pi05_find_hidden.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_ft_vlabench_find_hidden_lora}"
POLICY_DIR="${POLICY_DIR:-}"
SLOTS="${SLOTS:-}"

if [[ -z "${POLICY_DIR}" ]]; then
    echo "[err] POLICY_DIR=/path/to/checkpoint/<step> is required"; exit 1
fi
POLICY_DIR="${POLICY_DIR%/}"
[[ -d "${POLICY_DIR}" ]] || { echo "[err] no such checkpoint: ${POLICY_DIR}"; exit 1; }

# outputs/eval_<exp>_<step>[_<slots>] — keeps every run instead of overwriting one dir.
_step="$(basename "${POLICY_DIR}")"
_exp="$(basename "$(dirname "${POLICY_DIR}")")"
_suffix=""; [[ -n "${SLOTS}" ]] && _suffix="_${SLOTS//,/+}"
# Horizon belongs in the directory name too: a sweep over EVAL_HORIZON would
# otherwise write every point to the same dir and keep only the last one.
# Only non-default horizons get a tag, so existing result paths stay stable.
EVAL_HORIZON="${EVAL_HORIZON:-5}"
[[ "${EVAL_HORIZON}" != "5" ]] && _suffix="${_suffix}_h${EVAL_HORIZON}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_${_exp}_${_step}${_suffix}}"

cd "${REPO_ROOT}"

# ---- mic camera: which camera the audio (u,v) is projected through -----------
# The binaural listener moved camera 2 -> 1 in the v3 rebuild
# (src/audio/oracle_sled.TASK_MIC_CAM). A checkpoint's audio labels were built
# with whatever mic camera its *training data* used, so evaluating a pre-v3
# checkpoint under cam 1 hands the policy a different (u,v) reference frame than
# it learned — the same class of train/eval mismatch as the base-frame bug.
#
# Rather than trust the caller to remember, infer it from the checkpoint itself:
# training writes its norm stats to <ckpt>/assets/<repo_id>/, and the repo_id
# identifies the dataset exactly. MIC_CAM overrides. Unknown repo_id fails
# closed — a silently-wrong mic camera is how experiments get invalidated.
MIC_CAM="${MIC_CAM:-}"
if [[ -z "${MIC_CAM}" ]]; then
    _asset_repo="$(cd "${POLICY_DIR}/assets" 2>/dev/null &&
                   find . -name norm_stats.json -printf '%h\n' 2>/dev/null | head -1 | sed 's|^\./||')"
    case "${_asset_repo}" in
        # Glob, not an exact match: derived v3 sets (_rxfix, and anything later)
        # share v3's scene and therefore its mic camera. Matching only the bare
        # name would drop them into the pre-v3 branch below and silently feed a
        # camera-2 (u,v) frame to a camera-1 policy.
        local/avla_find_hidden_v3*) MIC_CAM=1 ;;  # v3 and derivatives: mic = camera 1
        local/avla_find_hidden*)    MIC_CAM=2 ;;  # v1/v2/framefix/top/bottom: mic = camera 2
        *)
            echo "[err] cannot infer the mic camera for this checkpoint."
            echo "      assets repo_id = '${_asset_repo:-<none found>}'"
            echo "      Pass MIC_CAM=2 (pre-v3 training data) or MIC_CAM=1 (v3) explicitly."
            exit 1 ;;
    esac
    echo "[eval] mic camera ${MIC_CAM}  (inferred from checkpoint assets repo_id '${_asset_repo}')"
else
    echo "[eval] mic camera ${MIC_CAM}  (explicit MIC_CAM override)"
fi

echo "[eval] slots=${SLOTS:-all}  ckpt=${POLICY_DIR}"
echo "[eval] out=${EVAL_DIR}"

# Pass VLABENCH_HIDDEN_SLOT_LABEL only when restricting; leaving it unset is what
# makes the task sample all four slots itself.
slot_env=()
[[ -n "${SLOTS}" ]] && slot_env+=("VLABENCH_HIDDEN_SLOT_LABEL=${SLOTS}")

env "${slot_env[@]}" \
VLABENCH_MIC_CAM="${MIC_CAM}" \
REPO_ROOT="${REPO_ROOT}" \
TASK_NAME="find_hidden_object_open" \
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}" \
LOCAL=1 POLICY_CONFIG="${POLICY_CONFIG}" POLICY_DIR="${POLICY_DIR}" \
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}" \
AUDIO_MODE="slots_uv" EVAL_MODE="oracle" \
EVAL_N="${EVAL_N:-100}" EVAL_MAX_LEN="${EVAL_MAX_LEN:-250}" \
EVAL_HORIZON="${EVAL_HORIZON}" TOP_K="${TOP_K:-3}" \
EVAL_SAVE_VIDEO="${EVAL_SAVE_VIDEO:-1}" \
NOISE_AZ_STD="${NOISE_AZ_STD:-0}" NOISE_EL_STD="${NOISE_EL_STD:-0}" \
NOISE_CONF_MIN="${NOISE_CONF_MIN:-1.0}" NOISE_CONF_MAX="${NOISE_CONF_MAX:-1.0}" \
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0}" \
NOISE_ENERGY_STD="${NOISE_ENERGY_STD:-0}" \
EVAL_DIR="${EVAL_DIR}" \
bash sh/eval_pi05_audio.sh

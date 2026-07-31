#!/usr/bin/env bash
# Rebuild the find_hidden dataset (v3): 880 episodes, 220 per slot.
#
# What changed vs the v2 source (dataset_find_hidden_v2_src):
#
# 1. Degenerate episodes are gated out. The cabinets' slide damping is lowered
#    50 -> 1 so the oracle can pull a drawer open, which also means nothing
#    holds a drawer statically: the hidden object dropping into place during the
#    reset settle shoved its own drawer out in ~20-25 % of resets (worst
#    open_fraction 0.118, vs the 0.13 success threshold). When it hit the
#    *target* drawer the task succeeded before the expert moved, and the
#    generator wrote out a "success" containing only the 2 s stay-still prefix
#    — 10 of the 880 v2 episodes are like this (20-39 frames, <= 0.176 m of end
#    effector travel, vs >= 64 frames and >= 0.208 m for every real one). The
#    same drift also made an ajar drawer a *visual* giveaway of the answer.
#    Now: every drawer is force-closed and re-settled after reset (repairs all
#    of the observed cases), then the scene is verified and rejected if a drawer
#    is still ajar, the success condition is already met, or the object is not
#    hidden in the drawer its slot label names. A short / barely-moving episode
#    is rejected after the rollout.
#
# 2. The microphone moved from camera 2 to camera 1. Camera 2 sits at z=1.75,
#    well above both drawers, which bunches their elevations together; camera 1
#    is re-posed to (0, -1.05, 1.20) in camera_config.json so the two drawers
#    straddle it. On the projected (u, v) the SlotEncoder consumes, over the 880
#    v2 episodes: elevation d' 5.54 -> 7.72, azimuth d' 12.35 -> 12.96, still
#    0/880 off-screen. Camera 1's stock pose was NOT reused: it is off-centre to
#    the left (x=-0.775), which makes both cues *worse* (u 9.89, v 4.87) and
#    pushes 12/880 episodes off-screen entirely. Camera 1's image is not in
#    convert_hdf5_to_lerobot's DEFAULT_CAM_MAP, so the policy's visual inputs
#    (cameras 2/0/3) are byte-for-byte unchanged from the framefix set.
#
# 3. The oracle audio GT is snapshotted at the FIRST recorded frame instead of
#    after the expert finished. It used to describe where the object ended up
#    once the drawer had been pulled 4-10 cm out, while eval recomputes the
#    direction from the live env on every policy step — starting from the hidden
#    position. That was a train/eval mismatch on exactly the frames where the
#    policy has to choose a drawer.
#
# 4. --slim-hdf5: drops the observation streams nothing downstream reads (depth,
#    point clouds, robot_mask, and the image_0..3 duplicate of `rgb`). 163 MB ->
#    ~33 MB per episode, so the source set is ~29 GB instead of 143 GB. The
#    converter reads observation/rgb, observation/ee_state, action and meta_info.
#
# The robot base frame fix is UNCHANGED and still required: generation records
# actions relative to the robot's *default* base position while eval uses this
# scene's get_robot_frame_position() = (0, -0.7, 0.7), a 28.8 cm / 7.4 cm gap in
# y / z. convert_hdf5_to_lerobot.py's ROBOT_FRAME_POS re-expresses both state and
# action in eval's frame, and only fires with --use-real-state (USE_REAL_STATE=1
# below). See KISTI_pi0_STATUS.md. The scene is untouched here — same robot base,
# same cabinets — so that fix applies verbatim. Verified after conversion below.
#
# Stages: generate -> convert -> audit. Training is deliberately NOT run here.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
cd "${REPO_ROOT}"

PER_SLOT="${PER_SLOT:-220}"          # 4 slots x 220 = 880
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TASK="find_hidden_object_open"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_find_hidden_v3_src}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_find_hidden_v3_lerobot}"
# Distinct repo_id so a checkpoint's training data is identifiable from its
# config alone (local/avla_find_hidden{,_framefix} are already taken by older,
# differently-broken copies).
REPO_ID="${REPO_ID:-local/avla_find_hidden_v3}"

DO_GENERATE="${DO_GENERATE:-1}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_AUDIT="${DO_AUDIT:-1}"

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; exit 1; }

# ---------------------------------------------------------------------------
# [1] generate
# ---------------------------------------------------------------------------
if [[ "${DO_GENERATE}" == "1" ]]; then
    say "=== [1] generate ${PER_SLOT}/slot into ${GEN_ROOT} (MAX_PARALLEL=${MAX_PARALLEL})"
    avail_gb=$(df -BG --output=avail "${REPO_ROOT}" | tail -1 | tr -dc '0-9')
    need_gb=$(( PER_SLOT * 4 * 35 / 1000 + 5 ))
    say "disk: ${avail_gb} GB free, need ~${need_gb} GB (slim HDF5)"
    (( avail_gb > need_gb )) || die "not enough disk (${avail_gb} GB free, need ~${need_gb} GB)"

    GEN_ROOT="${GEN_ROOT}" \
    STAGING="${REPO_ROOT}/dataset_find_hidden_v3_staging" \
    LOGDIR="${REPO_ROOT}/outputs/gen_find_hidden_v3_logs" \
    PER_SLOT="${PER_SLOT}" MAX_PARALLEL="${MAX_PARALLEL}" SLIM_HDF5=1 \
        bash sh/gen_find_hidden_balance.sh \
        || say "WARN: generator returned non-zero (partial slots are still usable)"
fi

n_ep=$(ls "${GEN_ROOT}/${TASK}"/data_*.hdf5 2>/dev/null | wc -l)
say "generated episodes: ${n_ep}"
(( n_ep > 0 )) || die "no episodes in ${GEN_ROOT}/${TASK}"

# ---------------------------------------------------------------------------
# [2] convert -> LeRobot (with the base-frame fix via --use-real-state)
# ---------------------------------------------------------------------------
if [[ "${DO_CONVERT}" == "1" ]]; then
    say "=== [2] convert -> ${LEROBOT_DIR} (repo_id ${REPO_ID})"
    (( n_ep >= PER_SLOT * 4 )) || say "WARN: only ${n_ep}/$((PER_SLOT*4)) episodes; converting anyway"
    DO_GENERATE=0 DO_CONVERT=1 DO_NORM=0 DO_TRAIN=0 DO_EVAL=0 \
    GEN_ROOT="${GEN_ROOT}" SRC_DIR="${GEN_ROOT}/${TASK}" \
    LEROBOT_DIR="${LEROBOT_DIR}" REPO_ID="${REPO_ID}" \
    USE_REAL_STATE=1 \
        bash sh/train_pi0_find_hidden.sh || die "conversion returned non-zero"
fi

# ---------------------------------------------------------------------------
# [3] audit — the checks that would have caught the v2 defects
# ---------------------------------------------------------------------------
if [[ "${DO_AUDIT}" == "1" ]]; then
    say "=== [3] audit ${GEN_ROOT}/${TASK} + ${LEROBOT_DIR}"
    third_party/openpi/.venv/bin/python scripts/audit_find_hidden_dataset.py \
        --src-dir "${GEN_ROOT}/${TASK}" --lerobot-dir "${LEROBOT_DIR}" \
        || die "audit failed"
fi

say "=== DONE"
say "  HDF5    -> ${GEN_ROOT}/${TASK}"
say "  LeRobot -> ${LEROBOT_DIR}  (repo_id ${REPO_ID})"

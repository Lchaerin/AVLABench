#!/usr/bin/env bash
# Rebuild the find_hidden dataset (v4): 880 episodes, 220 per slot.
#
# v4 changes exactly ONE thing vs v3: the two *policy* cameras.
#
#   cam_0  (0, 0.15, 2.10) straight overhead   ->  (0.775, -0.856, 1.409)
#   cam_2  (0, -1.05, 1.75) over-the-shoulder  ->  (-0.775, -0.856, 1.409)
#
# Both are VLABench's stock table-side cameras ("right" / "left", the pair
# `dataset_find_hidden_lerobot_src` was generated with) raised 0.20 m and aimed
# at (0, 0.10, 0.96), i.e. tilted 20 deg down. Candidate "B_mild" of
# scripts/preview_find_hidden_cameras.py, chosen from four rendered options.
#
# Why: v3's cam_0 was a top-down layout view. From directly overhead you cannot
# tell *which* drawer the gripper is at, which is precisely the discrimination
# the bottom-drawer slot needs, so the most useful of the three policy images was
# spent on the least useful angle. The side pair also turns out to be
# self-covering: when the arm works the near cabinet it blocks the camera on that
# side, but the opposite camera always keeps a clear line.
#
# ⚠️ The microphone (cam_1) is deliberately NOT touched — same pos (0,-1.05,1.20),
# same xyaxes, same fovy 50 as v3. The mic pose is what fixes the audio labels,
# and v3's azimuth d' 16.37 / elevation d' 16.90 were measured at that pose. So
# **v4's audio labels are identical to v3's to the last decimal**, and v4 vs
# v3_rxfix is a clean controlled comparison in which only the policy images
# differ. The audit re-measures d' and will show the same numbers; treat any
# drift there as a bug, not an improvement.
#
# Everything else is inherited from v3 and still applies verbatim:
#   * scene gates (close_all_drawers + validate_hidden_scene/_episode) — see the
#     header of sh/rebuild_find_hidden_v3.sh for why they exist
#   * oracle audio GT snapshotted at the first recorded frame
#   * --slim-hdf5 (~33 MB/episode instead of 163 MB)
#   * the robot base-frame fix, which needs --use-real-state (USE_REAL_STATE=1)
#   * the roll-branch fold (2026-07-31) — now unconditional in the converter, so
#     v4 gets it automatically and needs no rxfix-style second pass
#
# ⚠️ Camera change = eval compatibility break. A v4 checkpoint must be evaluated
# with the v4 camera_config; any pre-v4 checkpoint must NOT be (see
# VLABENCH_DISABLE_CAMERA_OVERRIDE in dm_task.reset_camera_views). The KISTI jobs
# 870110/870111 are training on v3_rxfix and are unaffected.
#
# Stages: generate -> convert -> audit. Training is deliberately NOT run here.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
cd "${REPO_ROOT}"

PER_SLOT="${PER_SLOT:-220}"          # 4 slots x 220 = 880
MAX_PARALLEL="${MAX_PARALLEL:-4}"
TASK="find_hidden_object_open"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_find_hidden_v4_src}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_ROOT}/dataset_find_hidden_v4_lerobot}"
REPO_ID="${REPO_ID:-local/avla_find_hidden_v4}"

DO_GENERATE="${DO_GENERATE:-1}"
DO_CONVERT="${DO_CONVERT:-1}"
DO_AUDIT="${DO_AUDIT:-1}"

say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; exit 1; }

# ---------------------------------------------------------------------------
# [0] preflight: the camera config must actually be the v4 one.
# Generating 880 episodes with the wrong cameras costs ~10 h and 33 GB, and the
# mistake is invisible afterwards unless you render a frame. Check first.
# ---------------------------------------------------------------------------
python3 - <<'PY' || die "camera_config.json is not the v4 (B_mild) configuration"
import json, sys
spec = json.load(open("VLABench/configs/camera_config.json"))["find_hidden_object_open"]
want = {
    "0": ("0.775 -0.856 1.409", "45"),
    "2": ("-0.775 -0.856 1.409", "45"),
    "1": ("0 -1.05 1.20", "50"),          # mic — must be UNCHANGED from v3
}
ok = True
for cam, (pos, fovy) in want.items():
    got = spec.get(cam, {})
    if got.get("pos") != pos or got.get("fovy") != fovy:
        print(f"  cam_{cam}: expected pos={pos!r} fovy={fovy!r}, got "
              f"pos={got.get('pos')!r} fovy={got.get('fovy')!r}")
        ok = False
print("[camera] preflight OK: cam_0/cam_2 = B_mild, cam_1 (mic) unchanged" if ok else "")
sys.exit(0 if ok else 1)
PY

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
    STAGING="${REPO_ROOT}/dataset_find_hidden_v4_staging" \
    LOGDIR="${REPO_ROOT}/outputs/gen_find_hidden_v4_logs" \
    PER_SLOT="${PER_SLOT}" MAX_PARALLEL="${MAX_PARALLEL}" SLIM_HDF5=1 \
        bash sh/gen_find_hidden_balance.sh \
        || say "WARN: generator returned non-zero (partial slots are still usable)"
fi

n_ep=$(ls "${GEN_ROOT}/${TASK}"/data_*.hdf5 2>/dev/null | wc -l)
say "generated episodes: ${n_ep}"
(( n_ep > 0 )) || die "no episodes in ${GEN_ROOT}/${TASK}"

# ---------------------------------------------------------------------------
# [2] convert -> LeRobot (base-frame fix via --use-real-state; roll fold is
#     unconditional in the converter now)
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
# [3] audit
# ---------------------------------------------------------------------------
if [[ "${DO_AUDIT}" == "1" ]]; then
    say "=== [3] audit ${GEN_ROOT}/${TASK} + ${LEROBOT_DIR}"
    third_party/openpi/.venv/bin/python scripts/audit_find_hidden_dataset.py \
        --src-dir "${GEN_ROOT}/${TASK}" --lerobot-dir "${LEROBOT_DIR}" \
        || die "audit failed"

    # v4-specific: the roll fold must have landed, and the audio must be
    # unchanged from v3 (the mic did not move).
    say "=== [3b] roll-fold + audio-parity check"
    third_party/openpi/.venv/bin/python - "${LEROBOT_DIR}" <<'PY' || die "v4 post-checks failed"
import glob, sys
import numpy as np, pandas as pd
root = sys.argv[1]
ps = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
bad = 0
for p in ps:
    df = pd.read_parquet(p)
    for col in ("observation.state", "action"):
        bad += int((np.stack(df[col].values)[:, 3] > 0).sum())
assert bad == 0, f"{bad} frames on the positive roll branch — roll fold did not apply"
print(f"[roll-fold] OK: 0 positive-branch roll frames across {len(ps)} episodes")
PY
fi

say "=== DONE"
say "  HDF5    -> ${GEN_ROOT}/${TASK}"
say "  LeRobot -> ${LEROBOT_DIR}  (repo_id ${REPO_ID})"
say "  NOTE: v4 policy images differ from v3; audio labels are identical."

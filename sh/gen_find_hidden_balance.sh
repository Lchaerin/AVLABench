#!/usr/bin/env bash
# Balanced oracle-mode dataset generator for find_hidden_object_open.
#
# Generates PER_SLOT successful episodes for each of the 4 candidate slots
# {left,right}_{top,bottom}, forcing the slot via --target-position-label
# (VLABENCH_HIDDEN_SLOT_LABEL) so the final distribution is exact regardless of
# per-slot oracle success rate. The 4 jobs run in parallel; oracle mode => no
# SLED/GPU inference, only lightweight EGL rendering. Successful episodes are
# then merged (re-indexed) into GEN_ROOT/find_hidden_object_open/.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
TASK="find_hidden_object_open"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_find_hidden_lerobot_src}"
STAGING="${STAGING:-${REPO_ROOT}/dataset_find_hidden_staging}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/outputs/gen_find_hidden_logs}"

PER_SLOT="${PER_SLOT:-15}"                 # successful episodes per slot
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"          # concurrent slot jobs (GPU-bound)
SLOTS=(left_top right_top left_bottom right_bottom)

cd "${REPO_ROOT}"
rm -rf "${STAGING}"; mkdir -p "${STAGING}" "${LOGDIR}" "${GEN_ROOT}/${TASK}"

echo "==================== [gen] find_hidden_object_open balanced ===================="
echo "  PER_SLOT=${PER_SLOT}  slots=${SLOTS[*]}"

echo "  MAX_PARALLEL=${MAX_PARALLEL} concurrent jobs (GPU-bound)"
fail=0
for slot in "${SLOTS[@]}"; do
    # throttle: wait until fewer than MAX_PARALLEL jobs are running
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n 2>/dev/null || fail=1
    done
    # bottom slots have a lower oracle success rate -> more attempts
    case "$slot" in
        *_bottom) mult=8 ;;
        *)        mult=3 ;;
    esac
    attempts=$(( PER_SLOT * mult + 20 ))
    save_dir="${STAGING}/${slot}"          # data -> save_dir/${TASK}/
    log="${LOGDIR}/${slot}.log"
    echo "  start ${slot}: need=${PER_SLOT} attempts<=${attempts}  log=${log}"
    VLABENCH_HIDDEN_SLOT_LABEL="${slot}" \
    conda run -n "${CONDA_ENV}" --no-capture-output python scripts/trajectory_generation.py \
        --task-name "${TASK}" --oracle-mode \
        --target-position-label "${slot}" \
        --save-dir "${save_dir}" \
        --n-sample "${attempts}" --max-episode "${PER_SLOT}" \
        --start-idle-seconds "${START_IDLE_SECONDS}" \
        --dataset-fps "${DATASET_FPS}" \
        >"${log}" 2>&1 &
done

echo "  all slots queued; waiting for completion..."
wait || fail=1
echo "==================== [gen] jobs finished (fail=${fail}) ===================="

# ---- merge staged episodes into GEN_ROOT/<task> with fresh indices ---------
echo "==================== [merge] into ${GEN_ROOT}/${TASK} ===================="
python3 - "$GEN_ROOT/$TASK" "$STAGING" "$TASK" "${SLOTS[@]}" <<'PY'
import os, glob, re, shutil, sys
dst, staging, task = sys.argv[1], sys.argv[2], sys.argv[3]
slots = sys.argv[4:]
os.makedirs(dst, exist_ok=True)
def max_idx(d):
    mx = -1
    for f in glob.glob(os.path.join(d, "data_*.hdf5")):
        m = re.search(r"data_(\d+)\.hdf5$", f)
        if m: mx = max(mx, int(m.group(1)))
    return mx
nxt = max_idx(dst) + 1
per_slot = {}
for slot in slots:
    src = os.path.join(staging, slot, task)
    files = sorted(glob.glob(os.path.join(src, "data_*.hdf5")),
                   key=lambda f: int(re.search(r"data_(\d+)", f).group(1)))
    per_slot[slot] = len(files)
    for f in files:
        i = int(re.search(r"data_(\d+)", f).group(1))
        shutil.copy(f, os.path.join(dst, f"data_{nxt}.hdf5"))
        for aux in (f"audio_meta_{i}.json", f"demo_{i}_success_True.mp4"):
            ap = os.path.join(src, aux)
            if os.path.exists(ap):
                newaux = aux.replace(f"_{i}", f"_{nxt}", 1) if "audio_meta" in aux \
                         else f"demo_{nxt}_success_True.mp4"
                shutil.copy(ap, os.path.join(dst, newaux))
        nxt += 1
print("[merge] per-slot episodes:", per_slot)
print("[merge] total episodes now in", dst, "=", max_idx(dst) + 1)
PY

echo "==================== [done] ===================="
echo "  dataset src (hdf5) -> ${GEN_ROOT}/${TASK}"
echo "  next: convert to LeRobot format (convert_hdf5_to_lerobot.py --oracle-mode)"

#!/usr/bin/env bash
# Balanced top-up generator for the 3 SELD-UV radio tasks.
#
# Goal: bring each task's TARGET-direction distribution to TARGET_PER_POS
# episodes per {left, middle, right} (~3*TARGET_PER_POS total per task),
# reusing whatever is already in dataset_seld_uv/<task>/ and only generating
# the per-direction deficit. Direction is forced via
# VLABENCH_TARGET_POSITION_LABEL (--target-position-label) so the final
# distribution is exact regardless of per-direction failure rates.
#
# All 9 (task x direction) jobs run as separate parallel processes. oracle
# mode => no SLED/GPU inference, only lightweight EGL rendering.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_seld_uv}"
STAGING="${STAGING:-${REPO_ROOT}/dataset_seld_uv_topup}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/outputs/gen_seld_uv_balance_logs}"

TARGET_PER_POS="${TARGET_PER_POS:-134}"   # ~402 per task total
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"

# Space-separated task list; default = all 3 SELD-UV radio tasks.
read -ra TASKS <<< "${TASKS:-select_radio select_radio_two select_radio_silent}"
POSITIONS=(left middle right)

cd "${REPO_ROOT}"
rm -rf "${STAGING}"; mkdir -p "${STAGING}" "${LOGDIR}"

# ---- helper: count existing episodes per direction (from audio_meta) -------
count_pos() {  # $1=task $2=pos  -> echoes existing count
    python3 - "$GEN_ROOT/$1" "$2" <<'PY'
import json, glob, sys
d, pos = sys.argv[1], sys.argv[2]
n = 0
for f in glob.glob(d + "/audio_meta_*.json"):
    try:
        if json.load(open(f)).get("position_label") == pos:
            n += 1
    except Exception:
        pass
print(n)
PY
}

echo "==================== [plan] balanced top-up ===================="
declare -A DEFICIT
for task in "${TASKS[@]}"; do
    line="  ${task}:"
    for pos in "${POSITIONS[@]}"; do
        have=$(count_pos "$task" "$pos")
        need=$(( TARGET_PER_POS - have ))
        (( need < 0 )) && need=0
        DEFICIT["${task}__${pos}"]=$need
        line+=" ${pos}=${have}->+${need}"
    done
    echo "$line"
done
echo "  target per direction = ${TARGET_PER_POS}"

# ---- launch 9 parallel jobs ------------------------------------------------
echo "==================== [gen] launching parallel jobs ===================="
PIDS=()
for task in "${TASKS[@]}"; do
    for pos in "${POSITIONS[@]}"; do
        need=${DEFICIT["${task}__${pos}"]}
        if (( need <= 0 )); then
            echo "  skip ${task}/${pos} (already at target)"; continue
        fi
        save_dir="${STAGING}/${task}__${pos}"       # parent; data -> save_dir/<task>/
        attempts=$(( need * 4 + 40 ))               # headroom for failures
        log="${LOGDIR}/${task}__${pos}.log"
        echo "  start ${task}/${pos}: need=${need} attempts<=${attempts}  log=${log}"
        VLABENCH_TARGET_POSITION_LABEL="${pos}" \
        conda run -n "${CONDA_ENV}" --no-capture-output python scripts/trajectory_generation.py \
            --task-name "${task}" --oracle-mode \
            --target-position-label "${pos}" \
            --save-dir "${save_dir}" \
            --n-sample "${attempts}" --max-episode "${need}" \
            --start-idle-seconds "${START_IDLE_SECONDS}" \
            --dataset-fps "${DATASET_FPS}" \
            >"${log}" 2>&1 &
        PIDS+=($!)
    done
done

echo "  launched ${#PIDS[@]} jobs; waiting..."
fail=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  [warn] pid $pid exited non-zero"; fail=1; }
done
echo "==================== [gen] all jobs finished (fail=${fail}) ===================="

# ---- merge staged episodes into GEN_ROOT/<task> with fresh indices ---------
echo "==================== [merge] into ${GEN_ROOT} ===================="
python3 - "$GEN_ROOT" "$STAGING" "${TASKS[@]}" <<'PY'
import os, glob, re, shutil, sys, json, collections
gen_root, staging = sys.argv[1], sys.argv[2]
tasks = sys.argv[3:]
def max_idx(d):
    mx = -1
    for f in glob.glob(os.path.join(d, "data_*.hdf5")):
        m = re.search(r"data_(\d+)\.hdf5$", f)
        if m: mx = max(mx, int(m.group(1)))
    return mx
for task in tasks:
    dst = os.path.join(gen_root, task)
    os.makedirs(dst, exist_ok=True)
    nxt = max_idx(dst) + 1
    added = 0
    for pos_parent in sorted(glob.glob(os.path.join(staging, f"{task}__*"))):
        src = os.path.join(pos_parent, task)
        for dfile in sorted(glob.glob(os.path.join(src, "data_*.hdf5")),
                            key=lambda p: int(re.search(r"data_(\d+)", p).group(1))):
            orig = int(re.search(r"data_(\d+)", dfile).group(1))
            shutil.copy2(dfile, os.path.join(dst, f"data_{nxt}.hdf5"))
            meta = os.path.join(src, f"audio_meta_{orig}.json")
            if os.path.exists(meta):
                shutil.copy2(meta, os.path.join(dst, f"audio_meta_{nxt}.json"))
            nxt += 1; added += 1
    # final distribution
    c = collections.Counter()
    for f in glob.glob(os.path.join(dst, "audio_meta_*.json")):
        try: c[json.load(open(f)).get("position_label")] += 1
        except Exception: pass
    total = len(glob.glob(os.path.join(dst, "data_*.hdf5")))
    print(f"  {task}: +{added} added -> {total} total  dist={dict(c)}")
PY

echo "==================== [done] balanced top-up ===================="
echo "  staging kept at ${STAGING} (rm -rf when satisfied)"
echo "  next: re-run pipeline convert+train (DO_GENERATE=0 DO_COMBINE=1 DO_CONVERT=1 ...)"

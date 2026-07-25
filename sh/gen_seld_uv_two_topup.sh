#!/usr/bin/env bash
# Balanced top-up generator for dataset_seld_uv/select_radio_two only.
#
# Adds TARGET_ADD new episodes on top of whatever already exists, balancing
# two axes at once:
#   1. Target-radio POSITION (left/middle/right) — success rate depends on
#      radio position, so we water-fill deficits toward an even 1/3 split
#      of the *final* total (forced via VLABENCH_TARGET_POSITION_LABEL, same
#      mechanism as sh/gen_seld_uv_balance.sh).
#   2. Target sound CLASS — raw sampling is file-count weighted (e.g.
#      Drums_Percussion/Speech dominate), so we water-fill the least-used
#      classes first (forced via the new VLABENCH_TARGET_CLASS_NAMES env,
#      consumed one class per successful episode inside
#      scripts/trajectory_generation.py).
#
# The ADD new episodes are split into 3 per-position jobs (one process per
# position, like the existing balance script), each given a pre-computed,
# shuffled list of target class names to cycle through — so only 3 parallel
# EGL jobs run, not one per (position, class) cell.
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
CONDA_ENV="${CONDA_ENV:-vlabench}"
TASK="select_radio_two"
GEN_ROOT="${GEN_ROOT:-${REPO_ROOT}/dataset_seld_uv}"
STAGING="${STAGING:-${REPO_ROOT}/dataset_seld_uv_two_topup}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/outputs/gen_seld_uv_two_topup_logs}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
SOUND_DIR="${SOUND_DIR:-/home/rllab/Desktop/crossCorr/soud_effects}"

TARGET_ADD="${TARGET_ADD:-300}"          # total new episodes to generate
START_IDLE_SECONDS="${START_IDLE_SECONDS:-2.0}"
DATASET_FPS="${DATASET_FPS:-10}"
SEED="${SEED:-0}"                        # deterministic class-list shuffle
POSITIONS=(left middle right)

cd "${REPO_ROOT}"
rm -rf "${STAGING}"; mkdir -p "${STAGING}" "${LOGDIR}"

# ---- plan: compute position + class deficits and per-position class lists -
PLAN_JSON="${LOGDIR}/plan.json"
python3 - "$GEN_ROOT/$TASK" "$TAXONOMY" "$SOUND_DIR" "$TARGET_ADD" "$SEED" "$PLAN_JSON" <<'PY'
import json, glob, sys, os, random, collections
import yaml

data_dir, taxonomy_path, sound_dir, target_add, seed, plan_path = sys.argv[1:7]
target_add = int(target_add)
random.seed(int(seed))
POSITIONS = ["left", "middle", "right"]

# ---- existing distribution ------------------------------------------------
pos_counts = collections.Counter()
class_counts = collections.Counter()
for f in glob.glob(os.path.join(data_dir, "audio_meta_*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    pos_counts[j.get("position_label")] += 1
    class_counts[j.get("class_name")] += 1
existing_total = sum(pos_counts.values())

# ---- classes that actually have wav files ----------------------------------
tax = yaml.safe_load(open(taxonomy_path))
classes = {int(k): v["name"] for k, v in tax["classes"].items()}
label_map = tax.get("label_map", {})

def folder_to_class(folder):
    if folder in label_map:
        return label_map[folder]
    s = folder.replace("_and_", ", ").replace("_", " ")
    if s in label_map:
        return label_map[s]
    sl = s.lower()
    for k, v in label_map.items():
        if k.lower() == sl:
            return v
    return -1

avail_classes = set()
for root, _, files in os.walk(sound_dir):
    folder = os.path.basename(root)
    cid = folder_to_class(folder)
    if cid >= 0 and any(f.lower().endswith(".wav") for f in files):
        avail_classes.add(classes[cid])

# ---- water-fill POSITION deficits: sum == target_add exactly ---------------
pos_alloc = {p: 0 for p in POSITIONS}
for _ in range(target_add):
    p = min(POSITIONS, key=lambda p: pos_counts[p] + pos_alloc[p])
    pos_alloc[p] += 1

# ---- water-fill CLASS deficits: sum == target_add exactly, favors the
# least-represented classes first ------------------------------------------
class_list_sorted = sorted(avail_classes)
class_alloc = {c: 0 for c in class_list_sorted}
for _ in range(target_add):
    c = min(class_list_sorted, key=lambda c: class_counts[c] + class_alloc[c])
    class_alloc[c] += 1

master_classes = []
for c, n in class_alloc.items():
    master_classes.extend([c] * n)
random.shuffle(master_classes)
assert len(master_classes) == target_add

# ---- split the shuffled class pool across the 3 position jobs -------------
plan = {}
idx = 0
for p in POSITIONS:
    need = pos_alloc[p]
    plan[p] = {
        "need": need,
        "classes": master_classes[idx: idx + need],
    }
    idx += need

with open(plan_path, "w") as f:
    json.dump(plan, f, indent=2)

print("==================== [plan] select_radio_two top-up ====================")
print(f"  existing total = {existing_total}  target_add = {target_add}  final = {existing_total + target_add}")
print(f"  existing position dist = {dict(pos_counts)}")
print(f"  position allocation     = {pos_alloc}")
print(f"  existing class dist ({len(class_counts)} classes) = {dict(sorted(class_counts.items(), key=lambda kv: -kv[1]))}")
print(f"  class allocation (top 10 by need) = {sorted(class_alloc.items(), key=lambda kv: -kv[1])[:10]}")
PY

# ---- launch 3 parallel jobs (one per position), each cycling its class list
echo "==================== [gen] launching parallel jobs ===================="
PIDS=()
for pos in "${POSITIONS[@]}"; do
    need=$(python3 -c "import json; print(json.load(open('${PLAN_JSON}'))['${pos}']['need'])")
    classes_csv=$(python3 -c "import json; print(','.join(json.load(open('${PLAN_JSON}'))['${pos}']['classes']))")
    if (( need <= 0 )); then
        echo "  skip ${pos} (need=0)"; continue
    fi
    save_dir="${STAGING}/${TASK}__${pos}"          # parent; data -> save_dir/<task>/
    attempts=$(( need * 4 + 40 ))                   # headroom for failures
    log="${LOGDIR}/${pos}.log"
    echo "  start ${pos}: need=${need} attempts<=${attempts}  log=${log}"
    VLABENCH_TARGET_POSITION_LABEL="${pos}" \
    VLABENCH_TARGET_CLASS_NAMES="${classes_csv}" \
    conda run -n "${CONDA_ENV}" --no-capture-output python scripts/trajectory_generation.py \
        --task-name "${TASK}" --oracle-mode \
        --target-position-label "${pos}" \
        --save-dir "${save_dir}" \
        --n-sample "${attempts}" --max-episode "${need}" \
        --start-idle-seconds "${START_IDLE_SECONDS}" \
        --dataset-fps "${DATASET_FPS}" \
        >"${log}" 2>&1 &
    PIDS+=($!)
done

echo "  launched ${#PIDS[@]} jobs; waiting..."
fail=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || { echo "  [warn] pid $pid exited non-zero"; fail=1; }
done
echo "==================== [gen] all jobs finished (fail=${fail}) ===================="

# ---- merge staged episodes into GEN_ROOT/select_radio_two with fresh indices
echo "==================== [merge] into ${GEN_ROOT}/${TASK} ===================="
python3 - "$GEN_ROOT" "$STAGING" "$TASK" <<'PY'
import os, glob, re, shutil, sys, json, collections
gen_root, staging, task = sys.argv[1], sys.argv[2], sys.argv[3]

def max_idx(d):
    mx = -1
    for f in glob.glob(os.path.join(d, "data_*.hdf5")):
        m = re.search(r"data_(\d+)\.hdf5$", f)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx

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
        nxt += 1
        added += 1

pos_c = collections.Counter()
cls_c = collections.Counter()
for f in glob.glob(os.path.join(dst, "audio_meta_*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    pos_c[j.get("position_label")] += 1
    cls_c[j.get("class_name")] += 1
total = len(glob.glob(os.path.join(dst, "data_*.hdf5")))
print(f"  {task}: +{added} added -> {total} total")
print(f"  final position dist = {dict(pos_c)}")
print(f"  final class dist ({len(cls_c)} classes) = {dict(sorted(cls_c.items(), key=lambda kv: -kv[1]))}")
PY

echo "==================== [done] select_radio_two top-up ===================="
echo "  staging kept at ${STAGING} (rm -rf when satisfied)"
echo "  next: re-run DO_GENERATE=0 DO_COMBINE=1 DO_CONVERT=1 ... sh/train_pi0_seld_uv_three_radio.sh"
echo "        (it combines dataset_seld_uv/{select_radio,select_radio_two,select_radio_silent})"

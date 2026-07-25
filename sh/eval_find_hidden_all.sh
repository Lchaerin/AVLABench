#!/usr/bin/env bash
# Evaluate several find_hidden_object_open checkpoints in parallel, throttled by
# free GPU memory so we never OOM whatever else is running (e.g. the pipeline's
# own 20k eval, or another training). Each checkpoint -> its own EVAL_DIR.
#
#   STEPS="5000 10000 15000 20000" MAX_PARALLEL=3 MIN_FREE_GB=11 \
#   bash sh/eval_find_hidden_all.sh
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
TASK="find_hidden_object_open"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_find_hidden_lora}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs/pi0_find_hidden_lora/${POLICY_CONFIG}/pi0_find_hidden_lora}"
LOGDIR="${LOGDIR:-${REPO_ROOT}/outputs/eval_find_hidden_logs}"

read -ra STEPS <<< "${STEPS:-5000 10000 15000 20000}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-11}"          # need this much free GPU before a launch

EVAL_N="${EVAL_N:-100}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-250}"
EVAL_HORIZON="${EVAL_HORIZON:-5}"
TOP_K="${TOP_K:-3}"
# clean oracle (train/eval parity), matching the radio pipelines
NOISE_AZ_STD="${NOISE_AZ_STD:-0}"; NOISE_EL_STD="${NOISE_EL_STD:-0}"
NOISE_CONF_MIN="${NOISE_CONF_MIN:-1.0}"; NOISE_CONF_MAX="${NOISE_CONF_MAX:-1.0}"
NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB:-0}"; NOISE_ENERGY_STD="${NOISE_ENERGY_STD:-0}"

cd "${REPO_ROOT}"
mkdir -p "${LOGDIR}"

free_gb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}'; }

echo "[eval-all] steps=${STEPS[*]}  MAX_PARALLEL=${MAX_PARALLEL}  MIN_FREE_GB=${MIN_FREE_GB}"
PIDS=()
for step in "${STEPS[@]}"; do
    POLICY_DIR="${CKPT_ROOT}/${step}"
    if [[ ! -d "${POLICY_DIR}" ]]; then
        echo "[eval-all] skip step ${step} (no checkpoint at ${POLICY_DIR})"; continue
    fi
    # throttle on job count
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do wait -n 2>/dev/null || true; done
    # throttle on free GPU memory
    while (( $(free_gb) < MIN_FREE_GB )); do
        echo "[eval-all] waiting for GPU (free=$(free_gb)GB < ${MIN_FREE_GB}GB) before step ${step}"
        sleep 30
    done
    EVAL_DIR="${REPO_ROOT}/outputs/eval_find_hidden_step${step}"
    log="${LOGDIR}/step${step}.log"
    echo "[eval-all] launch step ${step} -> ${EVAL_DIR}  (free=$(free_gb)GB)  log=${log}"
    (
        EVAL_DIR="${EVAL_DIR}" TASK_NAME="${TASK}" TAXONOMY="${TAXONOMY}" \
        EVAL_SAVE_VIDEO=1 \
        LOCAL=1 POLICY_CONFIG="${POLICY_CONFIG}" POLICY_DIR="${POLICY_DIR}" \
        OPENPI_ROOT="${OPENPI_ROOT}" AUDIO_MODE="slots_uv" EVAL_MODE="oracle" \
        EVAL_N="${EVAL_N}" EVAL_MAX_LEN="${EVAL_MAX_LEN}" EVAL_HORIZON="${EVAL_HORIZON}" \
        TOP_K="${TOP_K}" \
        NOISE_AZ_STD="${NOISE_AZ_STD}" NOISE_EL_STD="${NOISE_EL_STD}" \
        NOISE_CONF_MIN="${NOISE_CONF_MIN}" NOISE_CONF_MAX="${NOISE_CONF_MAX}" \
        NOISE_CLASS_FLIP_PROB="${NOISE_CLASS_FLIP_PROB}" \
        NOISE_ENERGY_STD="${NOISE_ENERGY_STD}" \
        bash sh/eval_pi05_audio.sh
    ) >"${log}" 2>&1 &
    PIDS+=($!)
    sleep 20   # stagger launches so model-load memory spikes don't collide
done

echo "[eval-all] all ${#PIDS[@]} evals launched; waiting..."
wait
echo "==================== [eval-all done] ===================="
# ---- aggregate per-checkpoint success (+ per-slot) --------------------------
python3 - "${REPO_ROOT}" "${STEPS[@]}" <<'PY'
import json, os, sys, glob
repo = sys.argv[1]; steps = sys.argv[2:]
def load_summary(d):
    for name in ("summary.json", "eval_summary.json", "results.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            try: return json.load(open(p))
            except Exception: return None
    # fallback: any *.json with success_rate
    for p in glob.glob(os.path.join(d, "*.json")):
        try:
            j = json.load(open(p))
            if isinstance(j, dict) and "success_rate" in j: return j
        except Exception: pass
    return None
print("\n=== find_hidden_object_open — success by checkpoint ===")
for s in steps:
    d = os.path.join(repo, f"outputs/eval_find_hidden_step{s}")
    j = load_summary(d)
    if j is None:
        print(f"  step {s:>6}: (no summary found in {d})"); continue
    sr = j.get("success_rate"); isr = j.get("intention_success_rate")
    line = f"  step {s:>6}: success={sr:.3f}" if isinstance(sr,(int,float)) else f"  step {s:>6}: success=?"
    if isinstance(isr,(int,float)): line += f"  intention={isr:.3f}"
    bp = j.get("by_position")
    if isinstance(bp, dict):
        parts = []
        for k,v in bp.items():
            r = v.get("rate") if isinstance(v,dict) else v
            parts.append(f"{k}={r:.2f}" if isinstance(r,(int,float)) else f"{k}={r}")
        line += "  [" + " ".join(parts) + "]"
    print(line)
print()
PY

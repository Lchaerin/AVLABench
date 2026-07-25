#!/usr/bin/env bash
# Audio-usage probe for the find_hidden pi0.5 policy: run the SAME checkpoint on
# the SAME (fixed-geometry) scenes under different audio manipulations and see if
# behaviour changes. If flipaz/zero produce the same endpoint distribution as
# `none`, the policy is NOT using the audio direction.
#
#   STEP=20000 EVAL_N=40 bash sh/ablate_find_hidden_audio.sh
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
TAXONOMY="${TAXONOMY:-${REPO_ROOT}/class_taxonomy.yaml}"
TASK="find_hidden_object_open"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_find_hidden_lora}"
STEP="${STEP:-20000}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs/pi0_find_hidden_lora/${POLICY_CONFIG}/pi0_find_hidden_lora}"
POLICY_DIR="${CKPT_ROOT}/${STEP}"
EVAL_N="${EVAL_N:-40}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-250}"
MODES=(none flipaz zero)
cd "${REPO_ROOT}"

echo "[ablate] checkpoint step=${STEP} dir=${POLICY_DIR}  N=${EVAL_N}  modes=${MODES[*]}"
[[ -d "${POLICY_DIR}" ]] || { echo "[err] no checkpoint at ${POLICY_DIR}"; exit 1; }

PIDS=()
for mode in "${MODES[@]}"; do
    EVAL_DIR="${REPO_ROOT}/outputs/ablate_find_hidden_${mode}"
    log="${REPO_ROOT}/outputs/ablate_${mode}.log"
    echo "[ablate] launch mode=${mode} -> ${EVAL_DIR}"
    (
        VLABENCH_EVAL_AUDIO_ABLATE="${mode}" \
        VLABENCH_HIDDEN_NO_JITTER=1 \
        EVAL_DIR="${EVAL_DIR}" TASK_NAME="${TASK}" TAXONOMY="${TAXONOMY}" \
        EVAL_SAVE_VIDEO=0 \
        LOCAL=1 POLICY_CONFIG="${POLICY_CONFIG}" POLICY_DIR="${POLICY_DIR}" \
        OPENPI_ROOT="${OPENPI_ROOT}" AUDIO_MODE="slots_uv" EVAL_MODE="oracle" \
        EVAL_N="${EVAL_N}" EVAL_MAX_LEN="${EVAL_MAX_LEN}" EVAL_HORIZON=5 TOP_K=3 \
        NOISE_AZ_STD=0 NOISE_EL_STD=0 NOISE_CONF_MIN=1.0 NOISE_CONF_MAX=1.0 \
        NOISE_CLASS_FLIP_PROB=0 NOISE_ENERGY_STD=0 \
        bash sh/eval_pi05_audio.sh
    ) >"${log}" 2>&1 &
    PIDS+=($!)
    sleep 20
done
echo "[ablate] launched ${#PIDS[@]} modes; waiting..."
wait
echo "==================== [ablate done] ===================="
python3 - "${REPO_ROOT}" "${MODES[@]}" <<'PY'
import json, os, sys, statistics as st, collections
repo=sys.argv[1]; modes=sys.argv[2:]
print("\n=== audio-usage probe: endpoint (EE final x) by GT azimuth side ===")
print("If the policy USES audio, 'none' separates left/right EE-x, 'flipaz' inverts it, 'zero' collapses.\n")
for m in modes:
    d=os.path.join(repo,f"outputs/ablate_find_hidden_{m}")
    p=os.path.join(d,"episodes.json")
    if not os.path.exists(p): print(f"  {m:7s}: (no episodes.json)"); continue
    eps=json.load(open(p))
    side=collections.defaultdict(list)
    succ=0
    for e in eps:
        pos=e.get("position_label") or ""
        fee=e.get("final_ee_pos")
        if fee is None: continue
        s = "left" if pos.startswith("left") else ("right" if pos.startswith("right") else "?")
        side[s].append(float(fee[0]))
        succ += 1 if e.get("success") else 0
    def m_(xs): return round(st.mean(xs),3) if xs else None
    lx=side.get("left",[]); rx=side.get("right",[])
    sep = (m_(lx)-m_(rx)) if lx and rx else None
    print(f"  {m:7s}: succ={succ}/{len(eps)}  EE-x left={m_(lx)} (n={len(lx)})  right={m_(rx)} (n={len(rx)})  L-R sep={sep}")
print()
PY

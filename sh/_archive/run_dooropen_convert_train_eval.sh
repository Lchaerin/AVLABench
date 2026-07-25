#!/usr/bin/env bash
# Stages 2-4 of the TEMPORARY door-open SmolVLA test.
#   2) convert dataset_dooropen -> LeRobot (reuses canonical sh, oracle)
#   3) train SmolVLA (reuses canonical sh, reduced steps)
#   4) QA the converted data + oracle eval with motion-vs-sound tracking
# Run AFTER sh/gen_dooropen_smolvla.sh has produced dataset_dooropen/.
set -euo pipefail

REPO_ROOT="/home/rllab/Desktop/AVLABench"
cd "${REPO_ROOT}"

DATASET_DIR="${REPO_ROOT}/dataset_dooropen"
LEROBOT_DIR="${REPO_ROOT}/dataset_dooropen_lerobot"
OUTPUT_DIR="${REPO_ROOT}/outputs/smolvla_dooropen"
EVAL_DIR="${REPO_ROOT}/outputs/eval_smolvla_dooropen"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
EVAL_N="${EVAL_N:-20}"
PY="conda run -n vlabench --no-capture-output python"

# Door-open simplified task + short chime window (must match generation).
export MUJOCO_GL=egl
export VLABENCH_MICROWAVE_DOOROPEN_ONLY=1
export VLABENCH_MICROWAVE_PREGRIP=0
export VLABENCH_MICROWAVE_MIN_DELAY_SEC=2.0
export VLABENCH_MICROWAVE_MAX_DELAY_SEC=6.0
export VLABENCH_MICROWAVE_REACT_WINDOW_SEC=15.0

n_h5=$(ls "${DATASET_DIR}/take_out_microwave_food"/data_*.hdf5 2>/dev/null | wc -l)
echo "[stage2-4] found ${n_h5} hdf5 episodes in ${DATASET_DIR}"
if (( n_h5 < 10 )); then echo "[err] too few episodes"; exit 1; fi

# ---- Stage 2+3: convert + train via the canonical recipe ------------------
echo "============================================================"
echo "[2-3] convert + train (steps=${TRAIN_STEPS})"
echo "============================================================"
# Skip convert if the LeRobot dataset is already built (DO_CONVERT=0 to force).
DO_CONVERT="${DO_CONVERT:-auto}"
if [[ "${DO_CONVERT}" == "auto" ]]; then
    if [[ -f "${LEROBOT_DIR}/meta/info.json" ]]; then DO_CONVERT=0; else DO_CONVERT=1; fi
fi
DATASET_DIR="${DATASET_DIR}" LEROBOT_DIR="${LEROBOT_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" \
  DO_GENERATE=0 DO_CONVERT="${DO_CONVERT}" DO_TRAIN=1 DO_EVAL=0 \
  USE_ORACLE=1 \
  MIN_DELAY_SEC=2.0 MAX_DELAY_SEC=6.0 REACT_WINDOW_SEC=15.0 \
  TRAIN_STEPS="${TRAIN_STEPS}" BATCH_SIZE=32 NUM_WORKERS=8 \
  bash sh/train_smolvla_take_out_microwave_food.sh

# ---- Stage 3.5: QA the converted data (did demos wait for the chime?) -----
echo "============================================================"
echo "[3.5] dataset QA: motion onset vs audio onset"
echo "============================================================"
${PY} scripts/qa_dataset_motion_vs_sound.py --lerobot-dir "${LEROBOT_DIR}" || true

# ---- Stage 4: oracle eval with motion-vs-sound tracking -------------------
CKPT="$(ls -t ${OUTPUT_DIR}/ckpt_step*.pt 2>/dev/null | head -1)"
if [[ -z "${CKPT}" ]]; then echo "[err] no ckpt in ${OUTPUT_DIR}"; exit 1; fi
echo "============================================================"
echo "[4] oracle eval (${EVAL_N} eps)  ckpt=${CKPT}"
echo "============================================================"
${PY} src/eval/eval_smolvla_audio.py \
    --ckpt "${CKPT}" \
    --pretrained lerobot/smolvla_vlabench \
    --taxonomy "${REPO_ROOT}/class_taxonomy.yaml" \
    --task-name take_out_microwave_food \
    --n-episodes "${EVAL_N}" \
    --max-episode-length 250 \
    --horizon 10 \
    --intention-threshold 0.1 \
    --intention-mode exclusive \
    --exclusive-intention-margin 0.03 \
    --save-dir "${EVAL_DIR}" \
    --oracle-mode \
    --noise-az-std 3.0 --noise-el-std 5.0 \
    --noise-conf-min 0.85 --noise-conf-max 0.98 --noise-class-flip-prob 0.02 \
    --class-smoothing-window 5 \
    --shuffle-audio-slots on \
    --canonicalize-audio-slots azimuth \
    --motion-thresholds 0.01,0.02,0.05,0.1,0.2 \
    --save-motion-trace \
    --save-video

# ---- Stage 4.5: plot motion vs sound -------------------------------------
python3 scripts/plot_motion_vs_sound.py --eval-dir "${EVAL_DIR}" \
    --thresholds 0.01,0.02,0.05,0.1,0.2 || true

echo "[done] eval -> ${EVAL_DIR}"

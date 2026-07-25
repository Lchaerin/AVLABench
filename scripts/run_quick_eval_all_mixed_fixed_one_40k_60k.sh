#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/rllab/Desktop/AVLABench"
PY="conda run -n vlabench --no-capture-output python"
PRETRAINED="lerobot/smolvla_vlabench"
TAXONOMY="${REPO_ROOT}/class_taxonomy.yaml"
OUT="${REPO_ROOT}/outputs/smolvla_oracle_all_mixed_fixed_one_nl_azcanon_lora_80k_v1"
LOG_DIR="/tmp/quick_eval_all_mixed_fixed_one_40k_60k_logs"
MAX_PARALLEL="${MAX_PARALLEL:-3}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

run_eval() {
    local step="$1"
    local task="$2"
    local suffix="$3"
    local ckpt="${OUT}/ckpt_step${step}.pt"
    local save_dir="${REPO_ROOT}/outputs/eval_quick50_smolvla_oracle_all_mixed_fixed_one_nl_azcanon_lora_${step}_${suffix}"
    local log="${LOG_DIR}/${step}_${suffix}.log"

    if [[ ! -f "${ckpt}" ]]; then
        echo "[err] missing checkpoint: ${ckpt}" >&2
        return 1
    fi

    {
        echo "============================================================"
        echo "quick eval ${task} ckpt_step${step}"
        echo "ckpt=${ckpt}"
        echo "out =${save_dir}"
        echo "============================================================"
        ${PY} src/eval/eval_smolvla_audio.py \
            --ckpt "${ckpt}" \
            --pretrained "${PRETRAINED}" \
            --taxonomy "${TAXONOMY}" \
            --task-name "${task}" \
            --n-episodes 50 \
            --max-episode-length 200 \
            --horizon 5 \
            --intention-threshold 0.1 \
            --intention-mode exclusive \
            --exclusive-intention-margin 0.03 \
            --save-dir "${save_dir}" \
            --oracle-mode \
            --noise-az-std 3.0 \
            --noise-el-std 5.0 \
            --noise-conf-min 0.85 \
            --noise-conf-max 0.98 \
            --noise-class-flip-prob 0.02 \
            --class-smoothing-window 5 \
            --shuffle-audio-slots on \
            --target-first-audio-slots off \
            --canonicalize-audio-slots azimuth
    } > "${log}" 2>&1
}

jobs_running=0
for spec in \
    "0040000 select_radio select_radio" \
    "0040000 select_radio_two select_radio_two" \
    "0040000 select_radio_silent select_radio_silent" \
    "0060000 select_radio select_radio" \
    "0060000 select_radio_two select_radio_two" \
    "0060000 select_radio_silent select_radio_silent"
do
    read -r step task suffix <<< "${spec}"
    run_eval "${step}" "${task}" "${suffix}" &
    jobs_running=$((jobs_running + 1))
    if (( jobs_running >= MAX_PARALLEL )); then
        wait -n
        jobs_running=$((jobs_running - 1))
    fi
done
wait

echo "============================================================"
echo "quick evals finished"
echo "logs: ${LOG_DIR}"
echo "============================================================"

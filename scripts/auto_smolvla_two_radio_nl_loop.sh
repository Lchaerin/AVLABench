#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
cd "${REPO_ROOT}"

TARGET_RATE="${TARGET_RATE:-0.40}"
LOG_DIR="${LOG_DIR:-/tmp}"
BASE_ENV=(
  DO_GENERATE=0
  DO_CONVERT=0
  DO_TRAIN=1
  DO_EVAL=1
  EVAL_N=100
  EVAL_SAVE_VIDEO=0
  AUDIO_FUSION_MODE=natural_language
  TARGET_FIRST_SLOTS=off
  TARGET_FIRST_AUDIO_SLOTS_EVAL=off
  SHUFFLE_SLOTS=on
  SHUFFLE_AUDIO_SLOTS_EVAL=on
  CANONICALIZE_SLOTS=azimuth
  CANONICALIZE_AUDIO_SLOTS_EVAL=azimuth
  TARGET_SLOT_PROTECT=on
  TRAIN_STEPS=40000
  SAVE_EVERY=4000
  LOG_EVERY=200
)

rate_of() {
  local summary="$1"
  python - "$summary" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    print(float(json.load(f)["success_rate"]))
PY
}

meets_target() {
  local summary="$1"
  python - "$summary" "$TARGET_RATE" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    rate = float(json.load(f)["success_rate"])
target = float(sys.argv[2])
raise SystemExit(0 if rate >= target else 1)
PY
}

meets_target_status() {
  local summary="$1"
  local ok
  ok="$(python - "${summary}" "${TARGET_RATE}" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    rate = float(json.load(f)["success_rate"])
target = float(sys.argv[2])
print("1" if rate >= target else "0")
PY
)"
  [[ "${ok}" == "1" ]]
}

wait_for_current_if_any() {
  local summary="$1"
  local pattern="$2"
  while [[ ! -f "${summary}" ]] && pgrep -af "${pattern}" >/dev/null; do
    date '+%F %T'
    find outputs/smolvla_oracle_two_nl_signfix_40k_v2 -maxdepth 1 -name 'ckpt_step*.pt' \
      -printf '%f\n' 2>/dev/null | sort | tail -5 || true
    sleep 900
  done
}

run_variant() {
  local name="$1"; shift
  local output="${REPO_ROOT}/outputs/${name}"
  local eval_dir="${REPO_ROOT}/outputs/eval_${name}"
  local log="${LOG_DIR}/${name}.log"
  rm -rf "${output}" "${eval_dir}"
  echo "[run] ${name}"
  env "${BASE_ENV[@]}" OUTPUT_DIR="${output}" EVAL_DIR="${eval_dir}" "$@" \
    bash sh/train_smolvla_oracle_two_radios.sh 2>&1 | tee "${log}"
  local summary="${eval_dir}/eval_summary.json"
  if [[ ! -f "${summary}" ]]; then
    echo "[err] missing summary: ${summary}" >&2
    return 1
  fi
  echo "[result] ${name} success_rate=$(rate_of "${summary}")"
  meets_target "${summary}"
}

main() {
  local current_summary="${REPO_ROOT}/outputs/eval_smolvla_oracle_two_nl_signfix_40k_v2/eval_summary.json"
  wait_for_current_if_any \
    "${current_summary}" \
    "smolvla_oracle_two_nl_signfix_40k_v2|eval_smolvla_oracle_two_nl_signfix_40k_v2"

  if [[ -f "${current_summary}" ]]; then
    echo "[result] smolvla_oracle_two_nl_signfix_40k_v2 success_rate=$(rate_of "${current_summary}")"
    if meets_target_status "${current_summary}"; then
      exit 0
    fi
  else
    echo "[warn] current sign-fixed natural-language summary not found; starting fallback variants"
  fi

  set +e
  run_variant smolvla_oracle_two_nl_azcanon_lm6_40k_v5 \
    AUDIO_MAX_LEN=128 UNFREEZE_LM_LAYERS=6 BATCH_SIZE=24 NUM_WORKERS=8 \
    IMAGE_COLOR_JITTER=0.05 IMAGE_TRANSLATE_PX=2 STATE_NOISE_STD=0.003 \
    DIRECTION_DROPOUT=0.05 AUDIO_CONF_DROPOUT=0.0
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then exit 0; fi

  set +e
  run_variant smolvla_oracle_two_nl_azcanon_lora_40k_v6 \
    AUDIO_MAX_LEN=128 UNFREEZE_LM_LAYERS=0 VLM_LORA=1 LORA_R=16 LORA_ALPHA=32 \
    LORA_DROPOUT=0.05 LORA_TARGET_MODULES=q_proj,v_proj \
    BATCH_SIZE=24 NUM_WORKERS=8 \
    IMAGE_COLOR_JITTER=0.05 IMAGE_TRANSLATE_PX=2 STATE_NOISE_STD=0.003 \
    DIRECTION_DROPOUT=0.05 AUDIO_CONF_DROPOUT=0.0
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then exit 0; fi

  set +e
  run_variant smolvla_oracle_two_class_tokens_azcanon_lm6_40k_v7 \
    AUDIO_FUSION_MODE=class_tokens AUDIO_MAX_LEN=96 UNFREEZE_LM_LAYERS=6 \
    CLASS_TOKEN_SCALE=0.2 BATCH_SIZE=24 NUM_WORKERS=8 \
    IMAGE_COLOR_JITTER=0.05 IMAGE_TRANSLATE_PX=2 STATE_NOISE_STD=0.003 \
    DIRECTION_DROPOUT=0.05 AUDIO_CONF_DROPOUT=0.0
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then exit 0; fi

  echo "[done] all configured variants stayed below target ${TARGET_RATE}" >&2
  exit 1
}

main "$@"

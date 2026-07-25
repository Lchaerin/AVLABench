#!/usr/bin/env bash
# Evaluate an audio-conditioned pi0/openpi policy locally by default.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_primitive}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi0_audio}"

REPO_ROOT="${REPO_ROOT}" \
POLICY_CONFIG="${POLICY_CONFIG}" \
EVAL_DIR="${EVAL_DIR}" \
bash "${REPO_ROOT}/sh/eval_pi05_audio.sh"


#!/usr/bin/env bash
# Generic wrapper for audio-conditioned openpi local evaluation.
# BACKBONE=pi05 uses pi05_ft_vlabench_primitive.
# BACKBONE=pi0  uses pi0_ft_vlabench_primitive.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
BACKBONE="${BACKBONE:-pi05}"

case "${BACKBONE}" in
    pi05|pi0.5)
        POLICY_CONFIG="${POLICY_CONFIG:-pi05_ft_vlabench_primitive}"
        EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi05_audio}"
        ;;
    pi0)
        POLICY_CONFIG="${POLICY_CONFIG:-pi0_ft_vlabench_primitive}"
        EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/outputs/eval_pi0_audio}"
        ;;
    *)
        echo "[err] BACKBONE must be pi05, pi0.5, or pi0"
        exit 1
        ;;
esac

REPO_ROOT="${REPO_ROOT}" \
POLICY_CONFIG="${POLICY_CONFIG}" \
EVAL_DIR="${EVAL_DIR}" \
bash "${REPO_ROOT}/sh/eval_pi05_audio.sh"


#!/usr/bin/env bash
# Build the openpi uv environment used by local pi0/pi0.5 episode evaluation.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rllab/Desktop/AVLABench}"
OPENPI_ROOT="${OPENPI_ROOT:-${REPO_ROOT}/third_party/openpi}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_CACHE_DIR

if [[ ! -f "${OPENPI_ROOT}/pyproject.toml" ]]; then
    echo "[err] openpi checkout not found at ${OPENPI_ROOT}"
    echo "      Run: git submodule update --init --recursive third_party/openpi"
    exit 1
fi

cd "${OPENPI_ROOT}"

uv sync
uv pip install --no-deps -e "${REPO_ROOT}"
# Keep openpi's pinned JAX/Flax/Optax stack intact. Install only AVLABench
# runtime extras that are not part of openpi instead of syncing the full
# examples/vlabench requirements file, which pins older transitive deps.
uv pip install mediapy open3d colorlog colorama gdown pynput peft
uv sync
uv pip install --no-deps -e "${REPO_ROOT}"

echo "[done] openpi local environment is ready"
echo "       test: uv --project ${OPENPI_ROOT} run python -c 'from openpi.training import config; print(config.get_config(\"pi05_ft_vlabench_primitive\").name)'"

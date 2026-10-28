#!/usr/bin/env bash
# TriForge environment setup for Ubuntu on WSL2 (recommended) or native Linux.
# Prerequisite on Windows: a current NVIDIA driver. Do NOT install a CUDA driver inside WSL;
# the Windows driver provides GPU passthrough. Check with `nvidia-smi` inside WSL first.
#
#   bash scripts/install_wsl.sh            # creates .venv with Python 3.12 via uv, CUDA 12.8 torch
#   CUDA_TAG=cu126 bash scripts/install_wsl.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_TAG="${CUDA_TAG:-cu128}"
PY="${PY:-3.12}"
# The CUDA wheels are large (cuBLAS alone is ~600 MB) and uv's default 30 s read timeout fails on slow
# links ("operation timed out" from pypi.nvidia.com). Allow long reads and retry the torch step.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-8}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --allow-existing --python "$PY" .venv
source .venv/bin/activate
for attempt in 1 2 3; do
  uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch && break
  [ "$attempt" = 3 ] && { echo "torch download failed 3 times; see docs/INSTALL.md (slow downloads)"; exit 1; }
  echo "torch download failed (attempt $attempt), retrying in 10 s; finished wheels are cached"; sleep 10
done
uv pip install -r requirements/gpu.txt
uv pip install -e ".[hf,serve,dev]"
python scripts/check_env.py
echo "Done. Activate with: source .venv/bin/activate"

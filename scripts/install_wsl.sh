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
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv venv --python "$PY" .venv
source .venv/bin/activate
uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" torch
uv pip install -r requirements/gpu.txt
uv pip install -e ".[hf,serve,dev]"
python scripts/check_env.py
echo "Done. Activate with: source .venv/bin/activate"

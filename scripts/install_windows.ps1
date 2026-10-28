# TriForge on native Windows (PowerShell). WSL2 is recommended (bitsandbytes and Triton are Linux-first);
# this path works for inference, evaluation and fp/adamw training.
#
# Never install into Anaconda *base*: conda's intel-openmp/MKL and pip's torch each ship
# libiomp5md.dll and the process aborts with "OMP: Error #15". Use a dedicated env.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
$ErrorActionPreference = "Stop"
$envName = if ($env:TRIFORGE_ENV) { $env:TRIFORGE_ENV } else { "triforge" }
$cuda = if ($env:CUDA_TAG) { $env:CUDA_TAG } else { "cu128" }   # cu130 had cuDNN DLL mismatches on Windows (2026-10)
conda create -y -n $envName python=3.11
conda run -n $envName python -m pip install --index-url "https://download.pytorch.org/whl/$cuda" torch
conda run -n $envName python -m pip install -r requirements/base.txt -r requirements/hf.txt -r requirements/serve.txt
conda run -n $envName python -m pip install -e .
conda run -n $envName python scripts/check_env.py
Write-Host "Done. Run: conda activate $envName   (plain 'python' is still conda base)"

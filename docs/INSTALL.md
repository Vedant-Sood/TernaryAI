# Installation (Windows 11 + WSL2, RTX 4060 Laptop 8 GB)

Recommended: **Ubuntu on WSL2**. The GPU works there through the Windows driver, bitsandbytes
(8-bit AdamW) and Triton are Linux-first, and there is no OpenMP clash with Anaconda.

Measured on the target laptop (2026-10-26): RTX 4060 Laptop, 8188 MiB, Windows driver 610.60 (CUDA 13.3
UMD), WSL2 kernel 6.18, `nvidia-smi` works inside WSL, 23 GB RAM visible to WSL.

## A. WSL2 Ubuntu (recommended)

```bash
# in Ubuntu (WSL2). Do NOT install an NVIDIA driver inside WSL; the Windows driver is used.
nvidia-smi                                  # must list the RTX 4060
git clone https://github.com/Vedant-Sood/TernaryAI.git && cd TernaryAI
bash scripts/install_wsl.sh                 # uv + Python 3.12 venv, CUDA 12.8 torch, deps, then check_env
source .venv/bin/activate
python scripts/check_env.py                 # prints GPU, VRAM, bf16, packages; exit code 1 on problems
python -m pytest -q                         # 84 tests, ~20 s
```

Keep the repo inside the Linux filesystem (`~/TernaryAI`, not `/mnt/c/...`): I/O on `/mnt/c` is
several times slower and hurts data loading and checkpointing.

By default WSL2 gets about half of system RAM. For the long-running preset (optimizer state offloaded
to RAM, ~10 GB) raise it in `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=26GB
swap=16GB
```

Then run `wsl --shutdown` from PowerShell.

## B. Native Windows (inference, evaluation, small training)

```powershell
# NEVER install into Anaconda "base" (see OMP Error #15 below).
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1   # creates conda env "triforge"
conda activate triforge
python scripts\check_env.py
```

Or by hand:

```powershell
conda create -y -n triforge python=3.11
conda activate triforge
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements\base.txt -r requirements\hf.txt -r requirements\serve.txt -r requirements\dev.txt
pip install -e .
```

On Windows, `cu128` worked on the target laptop and `cu130` failed to import (mismatched cuDNN DLLs).
bitsandbytes on native Windows is not covered here; use `optim.name: adamw_cpu` or WSL2.

## Troubleshooting

### OMP Error #15

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

**Cause:** Anaconda base ships Intel OpenMP (`Library\bin\libiomp5md.dll`, from `intel-openmp`/`mkl`), and the
pip torch wheel bundles another copy (`site-packages\torch\lib\libiomp5md.dll`). Two OpenMP runtimes in
one process abort. On the target laptop, base also had the **CPU-only** torch build (`2.14.1+cpu`), so
it could not use the GPU anyway.

**Fix:** use a dedicated env (section A or B). Run `conda activate <env>` first: plain `python` in
PowerShell is still Anaconda base. Do **not** set `KMP_DUPLICATE_LIB_OK=TRUE`; it hides the clash and
can give wrong results. `scripts/check_env.py` detects both situations.

### `torch.cuda.is_available()` is False
* `python -c "import torch; print(torch.version.cuda)"` prints `None`: you have a CPU wheel; reinstall
  with `--index-url https://download.pytorch.org/whl/cu128`.
* In WSL: `nvidia-smi` must work; update the *Windows* NVIDIA driver; `wsl --update`.

### CUDA out of memory during training
The trainer catches it, saves an emergency checkpoint if it can, and writes
`<output_dir>/oom_report.json` with `torch.cuda.memory_summary()` and concrete advice. It **never** shrinks the
model on its own. Then:
1. `python scripts/profile_hardware.py --config <cfg>` shows the estimated breakdown and a safe setting.
2. Lower `runtime.micro_batch_size` and raise `runtime.grad_accum_steps` (same effective batch).
3. Shorten `runtime.seq_len` / use `runtime.seq_len_schedule`; set `runtime.loss_chunk: 1024`.
4. `optim.name: adamw8bit` or `adamw_cpu`; for 2B, fewer `model.trainable_layers` or `strategy: lora`.
5. Close other GPU apps (browsers with hardware acceleration use VRAM on a laptop).

### Hugging Face downloads fail
`python scripts/check_hf_auth.py`; gated or private repos need `hf auth login`. For offline use, download once with
`scripts/download_hf.py` and set `HF_HUB_OFFLINE=1`.

### Slow first step on GPU
cuDNN/cuBLAS autotuning and the first allocation take a few seconds; throughput in `metrics.jsonl` is
logged every `log_interval` steps and excludes that only after the first window.

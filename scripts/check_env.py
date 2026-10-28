"""CUDA / dependency compatibility check. Run this first on a new machine.

    python scripts/check_env.py            # human-readable report, exit code 1 if something critical fails
    python scripts/check_env.py --json
"""
import argparse
import importlib
import json
import os
import platform
import subprocess
import sys

import _common  # noqa: F401


def check():
    rep = {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
           "problems": [], "warnings": []}
    conda = os.environ.get("CONDA_PREFIX", "")
    rep["conda_prefix"] = conda or None
    if conda and os.path.basename(conda).lower() in ("anaconda3", "miniconda3") and platform.system() == "Windows":
        rep["warnings"].append("You are in the conda *base* env. pip torch + conda MKL there crashes with "
                               "'OMP: Error #15'. Create a clean env (docs/INSTALL.md).")
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        rep["warnings"].append("KMP_DUPLICATE_LIB_OK is set: it hides an OpenMP clash and can give wrong results.")
    # torch import in a subprocess: an OMP abort kills the process, so isolate it
    code = ("import torch, json; d = {'torch': torch.__version__, 'cuda_build': torch.version.cuda, "
            "'cuda': torch.cuda.is_available()}\n"
            "if d['cuda']:\n"
            "    p = torch.cuda.get_device_properties(0); f, t = torch.cuda.mem_get_info()\n"
            "    a = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16); (a @ a).sum().item()\n"
            "    d.update(gpu=p.name, cc=f'{p.major}.{p.minor}', vram_gib=round(t / 2**30, 2), free_gib=round(f / 2**30, 2),"
            " bf16=torch.cuda.is_bf16_supported(), cudnn=torch.backends.cudnn.version(),"
            " flash_sdp=torch.backends.cuda.flash_sdp_enabled())\n"
            "print(json.dumps(d))")
    env = {k: v for k, v in os.environ.items() if k != "KMP_DUPLICATE_LIB_OK"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout).strip().splitlines()
        if any("OMP: Error #15" in line for line in msg):
            rep["problems"].append("OMP Error #15: two OpenMP runtimes (conda intel-openmp + pip torch). "
                                   "Use a clean env; see docs/INSTALL.md#omp-error-15")
        else:
            rep["problems"].append("torch failed to import: " + (msg[-1] if msg else f"exit {r.returncode}"))
    else:
        rep["torch"] = json.loads(r.stdout.strip().splitlines()[-1])
        if not rep["torch"]["cuda"]:
            rep["warnings"].append("CUDA not available" + (" (CPU-only torch build)" if not rep["torch"]["cuda_build"]
                                                           else " (driver / WSL GPU passthrough?)"))
    for mod in ("numpy", "safetensors", "tokenizers", "yaml", "psutil", "transformers", "datasets",
                "huggingface_hub", "accelerate", "peft", "bitsandbytes", "triton", "fastapi", "uvicorn"):
        try:
            m = importlib.import_module(mod)
            rep.setdefault("packages", {})[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            rep.setdefault("packages", {})[mod] = None
            if mod in ("numpy", "safetensors", "tokenizers", "yaml"):
                rep["problems"].append(f"missing required package {mod} ({type(e).__name__})")
    if rep["packages"].get("bitsandbytes") is None:
        rep["warnings"].append("bitsandbytes missing: optim.name=adamw8bit unavailable (use adamw_cpu or install it)")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        rep["nvidia_smi"] = smi.stdout.strip() or smi.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        rep["nvidia_smi"] = None
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rep = check()
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        for k, v in rep.items():
            if k not in ("problems", "warnings"):
                print(f"{k:14s} {v}")
        for w in rep["warnings"]:
            print(f"WARNING  {w}")
        for e in rep["problems"]:
            print(f"PROBLEM  {e}")
        print("OK" if not rep["problems"] else "FAILED")
    sys.exit(1 if rep["problems"] else 0)


if __name__ == "__main__":
    main()

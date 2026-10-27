"""Train a TriForge 2B specialist on Modal (https://modal.com), with checkpoints on a Modal Volume.

    pip install modal && modal setup                      # once: sign in, creates the API token
    modal run --detach cloud/modal_train.py --specialist trimath --preset quick
    modal run --detach cloud/modal_train.py --specialist trimath --preset partial

Run from the repository root. The local checkout is copied into the image, so no GitHub token is
needed and uncommitted changes are included. `--detach` keeps the run going after the terminal closes;
follow it with `modal app logs triforge` or on the Modal dashboard.

Everything that must outlive a container lives on the Volume `triforge` (mounted at /vol):
checkpoints/, data/ (tokenized datasets) and hf/ (Hugging Face cache, so the 2B model is
downloaded once). A run longer than one function call (24 h limit) simply resumes on the next
`modal run`. Download results with `modal volume get triforge checkpoints/exports ./exports`.

GPU: L4 (24 GB, bf16) by default, using the bitnet2b_<preset>_l4.yaml presets. Set TRIFORGE_GPU=A10G
for an A10G (same presets), or TRIFORGE_GPU=T4 for a T4 (switches to the fp16 _t4 presets).
"""
import os
import pathlib
import subprocess

import modal

REPO = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/root/TernaryAI"
GPU = os.environ.get("TRIFORGE_GPU", "L4")
PROFILE = "t4" if GPU.upper().startswith("T4") else "l4"
TIMEOUT_H = 24  # Modal's maximum for one function call

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch>=2.6", "bitsandbytes>=0.43")
    .pip_install_from_requirements(str(REPO / "requirements" / "base.txt"))
    .pip_install_from_requirements(str(REPO / "requirements" / "hf.txt"))
    .env({"HF_HOME": "/vol/hf", "TRIFORGE_ROOT": REMOTE, "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO, REMOTE, copy=True,
                   ignore=[".git", ".venv", "checkpoints", "data", "runs", "results/**/*.jsonl",
                           "**/__pycache__", "**/*.egg-info"])
    .run_commands(f"pip install --no-deps -e {REMOTE}")
)
volume = modal.Volume.from_name("triforge", create_if_missing=True)
app = modal.App("triforge", image=image)


@app.function(gpu=GPU, volumes={"/vol": volume}, timeout=TIMEOUT_H * 3600)
def train(specialist: str, preset: str, profile: str, export: bool = True):
    # checkpoints/ and data/ of the checkout point at the Volume: the trainer writes there directly
    for d in ("checkpoints", "data"):
        os.makedirs(f"/vol/{d}", exist_ok=True)
        os.symlink(f"/vol/{d}", f"{REMOTE}/{d}")
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"])
    cmd = ["python", "scripts/cloud_session.py", "--specialist", specialist, "--preset", preset,
           "--profile", profile, "--input-root", "/nonexistent", "--budget-hours", str(TIMEOUT_H - 0.5)]
    if export:
        cmd.append("--export-when-done")
    try:
        rc = subprocess.call(cmd, cwd=REMOTE)
    finally:
        volume.commit()
    if rc != 0:
        raise RuntimeError(f"cloud_session.py exited with code {rc}")


@app.local_entrypoint()
def main(specialist: str = "trimath", preset: str = "quick", export: bool = True):
    print(f"TriForge on Modal: {specialist} / {preset} on {GPU} (configs bitnet2b_{preset}_{PROFILE}.yaml)")
    train.remote(specialist, preset, PROFILE, export)  # PROFILE is decided here: the env var is local

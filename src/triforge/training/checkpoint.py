"""Resumable checkpoints.

Layout of a run directory:
    <out>/config.yaml             resolved run config
    <out>/metrics.jsonl           one JSON line per logged step / eval
    <out>/ckpt/step_000500/       model.safetensors, optim.pt, state.json
    <out>/ckpt/latest             text file with the name of the newest complete checkpoint

Each checkpoint is written to a temporary directory and renamed only when complete, so an
interrupted save (power loss, Ctrl-C, OOM) never corrupts the newest usable checkpoint.
state.json records step, tokens seen, RNG states, data-sampler state and a SHA-256 of the
model file so integrity can be verified on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def model_tensors(model: torch.nn.Module) -> dict:
    """State dict with tied/shared tensors stored once (safetensors rejects shared storage)."""
    out, seen = {}, {}
    for k, v in model.state_dict().items():
        key = (v.data_ptr(), v.shape, v.dtype) if v.numel() else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen[key] = k
        out[k] = v.detach().contiguous().cpu()
    return out


def save_checkpoint(out_dir, step, model, optimizer=None, scheduler_state=None, extra=None, keep_last=3):
    ckdir = os.path.join(out_dir, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    name = f"step_{step:07d}"
    tmp = os.path.join(ckdir, f".tmp_{name}_{os.getpid()}")
    final = os.path.join(ckdir, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    mpath = os.path.join(tmp, "model.safetensors")
    save_file(model_tensors(model), mpath)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(tmp, "optim.pt"))
    torch.save(rng_state(), os.path.join(tmp, "rng.pt"))
    state = {"step": step, "time": time.time(), "model_sha256": _sha256(mpath),
             "scheduler": scheduler_state or {}, **(extra or {})}
    with open(os.path.join(tmp, "state.json"), "w") as f:
        json.dump(state, f, indent=2)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    with open(os.path.join(ckdir, "latest.tmp"), "w") as f:
        f.write(name)
    os.replace(os.path.join(ckdir, "latest.tmp"), os.path.join(ckdir, "latest"))
    # retention
    steps = sorted(d for d in os.listdir(ckdir) if d.startswith("step_"))
    for old in steps[:-keep_last] if keep_last else []:
        shutil.rmtree(os.path.join(ckdir, old), ignore_errors=True)
    return final


def latest_checkpoint(out_dir):
    p = os.path.join(out_dir, "ckpt", "latest")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = os.path.join(out_dir, "ckpt", f.read().strip())
    return d if os.path.exists(os.path.join(d, "state.json")) else None


def load_checkpoint(path, model, optimizer=None, map_location="cpu", verify=True, strict=True):
    with open(os.path.join(path, "state.json")) as f:
        state = json.load(f)
    mpath = os.path.join(path, "model.safetensors")
    if verify and _sha256(mpath) != state["model_sha256"]:
        raise RuntimeError(f"checkpoint {path} failed its SHA-256 integrity check")
    tensors = load_file(mpath, device=str(map_location))
    missing, unexpected = model.load_state_dict(tensors, strict=False)
    # tied weights are stored once, so their alias shows up as "missing"; anything else is an error
    tied = {k for k in missing if k.endswith("lm_head.weight") and getattr(model, "cfg", None)
            and getattr(model.cfg, "tie_embeddings", False)}
    real_missing = [k for k in missing if k not in tied]
    if strict and (real_missing or unexpected):
        raise RuntimeError(f"checkpoint mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}")
    if optimizer is not None and os.path.exists(os.path.join(path, "optim.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(path, "optim.pt"), map_location=map_location,
                                             weights_only=False))
    rng_path = os.path.join(path, "rng.pt")
    if os.path.exists(rng_path):
        set_rng_state(torch.load(rng_path, weights_only=False))
    return state

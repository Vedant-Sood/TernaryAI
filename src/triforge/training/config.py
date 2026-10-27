"""YAML run configuration with inheritance.

A run config has five independent sections plus metadata:

    inherit: [../base/runtime_rtx4060.yaml]     # optional, merged first (paths relative to this file)
    run:      {name, specialist, output_dir, seed}
    model:    architecture (TriForgeConfig fields) or {init_from: ..., strategy: ...}
    data:     dataset manifest(s) and mixture
    optim:    optimizer and schedule
    runtime:  device, precision, micro-batch, accumulation, checkpointing
    eval:     validation and benchmark settings

Later files override earlier ones key by key (deep merge). Command-line overrides use dotted
keys: `--set optim.lr=1e-3 runtime.micro_batch_size=2`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

SECTIONS = ("run", "model", "data", "optim", "runtime", "eval", "distill")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    merged: dict = {}
    for parent in cfg.pop("inherit", []) or []:
        merged = deep_merge(merged, load_config(path.parent / parent))
    merged = deep_merge(merged, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = merged
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    merged.setdefault("run", {}).setdefault("config_path", str(path))
    return merged


def resolve_path(p: str, root: str | None = None) -> str:
    """Paths in configs are relative to the repository root (or $TRIFORGE_ROOT)."""
    if p is None or os.path.isabs(p):
        return p
    root = root or os.environ.get("TRIFORGE_ROOT") or str(Path(__file__).resolve().parents[3])
    return os.path.join(root, p)


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

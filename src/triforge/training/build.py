"""Turn the `model` section of a run config into a model ready for training.

model.source:
  scratch      new TriForgeLM from `model.arch` (Phase A, small specialists)
  bitnet       BitNet b1.58 2B4T bf16 master weights (Phase C/D, ~2.4B params incl. embeddings)
  hf_dense     dense HF CausalLM, ternarized on load (Phase B conversion experiments)
  triforge     an earlier TriForge export / checkpoint directory (continue training)

model.strategy:
  full         every ternary layer trainable (latent weights + optimizer state for all)
  partial      only `trainable_layers` keep latent weights; other blocks are packed and frozen
  lora         ternary-preserving LoRA (delta added before ternarization, merged at export)
"""

from __future__ import annotations

import json
import os

from ..architectures import TriForgeConfig, TriForgeLM
from .config import resolve_path


def _layers(spec, n_layers):
    if spec is None:
        return None
    if isinstance(spec, str) and spec.startswith("last:"):
        k = int(spec.split(":")[1])
        return list(range(n_layers - k, n_layers))
    if isinstance(spec, str) and spec.startswith("every:"):
        k = int(spec.split(":")[1])
        return list(range(k - 1, n_layers, k))
    return list(spec)


def build_model(mcfg: dict, device="cpu"):
    source = mcfg.get("source", "scratch")
    strategy = mcfg.get("strategy", "full")
    lora_rank = mcfg.get("lora_rank", 0) if strategy == "lora" else 0
    if source == "scratch":
        arch = dict(mcfg["arch"])
        arch.setdefault("lora_rank", lora_rank)
        arch.setdefault("gradient_checkpointing", mcfg.get("gradient_checkpointing", False))
        model = TriForgeLM(TriForgeConfig.from_dict(arch))
    elif source == "bitnet":
        from ..huggingface.bitnet_weights import load_bitnet
        n_layers = mcfg.get("n_layers_hint", 30)
        layers = _layers(mcfg.get("trainable_layers"), n_layers) if strategy == "partial" else None
        model = load_bitnet(mcfg["path"], trainable_layers=layers, lora_rank=lora_rank,
                            max_seq_len=mcfg.get("max_seq_len", 4096), revision=mcfg.get("revision"),
                            gradient_checkpointing=mcfg.get("gradient_checkpointing", True),
                            train_embeddings=mcfg.get("train_embeddings", False),
                            cache_frozen_codes=mcfg.get("frozen_cache_codes", True))
    elif source == "hf_dense":
        from ..huggingface.hf_models import load_dense, ternarize_hf
        model = load_dense(mcfg["path"], revision=mcfg.get("revision"))
        if mcfg.get("ternarize", True):
            ternarize_hf(model, act_bits=mcfg.get("act_bits", 8), lora_rank=lora_rank)
        if mcfg.get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
    elif source == "triforge":
        from ..inference.export import load_export
        model = load_export(resolve_path(mcfg["path"]), trainable=True)
    else:
        raise ValueError(f"unknown model.source {source}")
    return model.to(device)


def trainable_summary(model) -> dict:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"trainable_params": tr, "parameters": total, "buffer_bytes": buffers}


def save_arch(model, out_dir):
    cfg = getattr(model, "cfg", None)
    if isinstance(cfg, TriForgeConfig):
        cfg.save(os.path.join(out_dir, "arch.json"))
    elif cfg is not None:
        with open(os.path.join(out_dir, "arch.json"), "w") as f:
            json.dump(cfg.__dict__, f, indent=2)


def load_teacher(dcfg: dict, device):
    if dcfg["teacher"]["source"] == "hf_dense":
        from ..huggingface.hf_models import load_dense
        t = load_dense(dcfg["teacher"]["path"], revision=dcfg["teacher"].get("revision"), device=device)
    else:
        t = build_model(dcfg["teacher"], device=device)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t



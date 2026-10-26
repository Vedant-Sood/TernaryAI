"""Export trained models to a compact, self-describing ternary format, and load them back.

Export directory:
    config.json            TriForgeConfig
    triforge.json          format version, packing, specialist, tokenizer/template, provenance, status
    model.safetensors      for each ternary linear: <name>.packed (uint8), <name>.scale (fp32)
                           everything else (embeddings, head, norms) in fp16/bf16
    tokenizer.json | tokenizer files

`status` must be one of: "trained", "untrained", "converted-not-finetuned", so an untrained or
merely converted model can never be mistaken for a trained one.
"""

from __future__ import annotations

import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

from ..architectures import TriForgeConfig, TriForgeLM
from ..quantization.bitlinear import BitLinear, PackedTernaryLinear
from ..quantization.packing import pack

FORMAT_VERSION = 1
STATUSES = ("trained", "untrained", "converted-not-finetuned")


@torch.no_grad()
def export_model(model: TriForgeLM, out_dir: str, *, status: str, specialist: str, tokenizer=None,
                 template="triforge", fmt="2bit", fp_dtype=torch.float16, provenance: dict | None = None):
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    os.makedirs(out_dir, exist_ok=True)
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.merge_lora()
    tensors, ternary_names = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            t, gamma = mod.ternary_weight()
            tensors[f"{name}.packed"] = pack(t.cpu(), fmt)
            tensors[f"{name}.scale"] = gamma.float().cpu().reshape(())
            ternary_names.append(name)
        elif isinstance(mod, PackedTernaryLinear):
            if mod.fmt != fmt:
                tensors[f"{name}.packed"] = pack(mod.codes().cpu(), fmt)
            else:
                tensors[f"{name}.packed"] = mod.packed.cpu()
            tensors[f"{name}.scale"] = mod.scale.float().cpu().reshape(())
            ternary_names.append(name)
    skip = {f"{n}.weight" for n in ternary_names} | {f"{n}.packed" for n in ternary_names} | \
           {f"{n}.scale" for n in ternary_names}
    for k, v in model.state_dict().items():
        if k in skip or k in tensors or "lora_" in k:
            continue
        if k == "lm_head.weight" and model.cfg.tie_embeddings:
            continue
        tensors[k] = v.detach().to(fp_dtype).contiguous().cpu() if v.is_floating_point() else v.cpu()
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    model.cfg.save(os.path.join(out_dir, "config.json"))
    meta = {"format": "triforge-ternary", "format_version": FORMAT_VERSION, "packing": fmt,
            "fp_dtype": str(fp_dtype).replace("torch.", ""), "specialist": specialist, "status": status,
            "template": template, "ternary_modules": len(ternary_names),
            "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "provenance": provenance or {}}
    if tokenizer is not None:
        tokenizer.save(out_dir)
        meta["tokenizer"] = {"kind": tokenizer.kind, "name": tokenizer.name}
    with open(os.path.join(out_dir, "triforge.json"), "w") as f:
        json.dump(meta, f, indent=2)
    size = os.path.getsize(os.path.join(out_dir, "model.safetensors"))
    return {"dir": out_dir, "bytes": size, "ternary_modules": len(ternary_names)}


@torch.no_grad()
def load_export(path: str, trainable=False, device="cpu", backend="reference", dtype=None) -> TriForgeLM:
    """Load an export. trainable=False -> PackedTernaryLinear (inference). True -> BitLinear latents."""
    cfg = TriForgeConfig.load(os.path.join(path, "config.json"))
    with open(os.path.join(path, "triforge.json")) as f:
        meta = json.load(f)
    tensors = load_file(os.path.join(path, "model.safetensors"))
    model = TriForgeLM(cfg)
    fmt = meta["packing"]
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if not isinstance(child, BitLinear):
                continue
            packed = tensors.pop(f"{full}.packed")
            scale = tensors.pop(f"{full}.scale")
            if trainable:
                from ..huggingface.bitnet_weights import latent_equivalent
                from ..quantization.packing import unpack
                codes = unpack(packed, (child.out_features, child.in_features), fmt)
                child.weight.copy_(latent_equivalent(codes, scale))
            else:
                p = PackedTernaryLinear(child.in_features, child.out_features, input_norm=child.norm,
                                        act_bits=child.act_bits, fmt=fmt, backend=backend)
                p.packed, p.scale = packed, scale.float()
                setattr(mod, cname, p)
    float_dtype = dtype or torch.float32
    missing, unexpected = model.load_state_dict({k: (v.to(float_dtype) if v.is_floating_point() else v)
                                                 for k, v in tensors.items()}, strict=False)
    allowed_missing = {"lm_head.weight"} if cfg.tie_embeddings else set()
    bad = [k for k in missing if k not in allowed_missing and not k.endswith((".packed", ".scale"))
           and not (trainable and k.endswith("proj.weight"))]
    if bad or unexpected:
        raise RuntimeError(f"export mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
    model.export_meta = meta
    if dtype is not None:
        for n, p in model.named_parameters():
            p.data = p.data.to(dtype)
    return model.to(device).eval() if not trainable else model.to(device)


def export_size_report(path: str) -> dict:
    total = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path))
    weights = os.path.getsize(os.path.join(path, "model.safetensors"))
    return {"export_dir_bytes": total, "model_file_bytes": weights}

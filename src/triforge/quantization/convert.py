"""Convert dense models to ternary-weight models.

`ternarize_linears` swaps nn.Linear modules for BitLinear, copying the dense weights in as
the latent weights. Right after conversion, the forward pass already uses ternary weights,
so this is post-training ternarization (PTQ). For dense pretrained models that usually
destroys quality (see docs/RESEARCH.md); quantization-aware fine-tuning and distillation
(triforge.distillation) are what can recover it. Conversion alone is not training.
"""

from __future__ import annotations

import fnmatch

import torch
import torch.nn as nn

from .bitlinear import BitLinear, PackedTernaryLinear
from .ternary import ternarize

DEFAULT_SKIP = ("*lm_head*", "*embed*", "*head")


def ternarize_linears(model: nn.Module, skip=DEFAULT_SKIP, act_bits=8, input_norm=False,
                      lora_rank=0, lora_alpha=None) -> list[str]:
    """In-place replacement of nn.Linear -> BitLinear. Returns the converted module names."""
    converted = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if type(child) is not nn.Linear or any(fnmatch.fnmatch(full, p) for p in skip):
                continue
            new = BitLinear(child.in_features, child.out_features, bias=child.bias is not None,
                            input_norm=input_norm, act_bits=act_bits, lora_rank=lora_rank,
                            lora_alpha=lora_alpha, device=child.weight.device, dtype=child.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(mod, child_name, new)
            converted.append(full)
    return converted


@torch.no_grad()
def ternary_report(model: nn.Module) -> dict:
    """Parameter accounting: how many weights are actually ternary in the forward pass."""
    n_ternary = n_zero = 0
    ternary_param_ids = set()
    for m in model.modules():
        if isinstance(m, BitLinear) and m.weight_quant_enabled:
            t, _ = m.ternary_weight()
            ternary_param_ids.add(id(m.weight))
            n_ternary += t.numel()
            n_zero += (t == 0).sum().item()
        elif isinstance(m, PackedTernaryLinear):
            t = m.codes()
            n_ternary += t.numel()
            n_zero += (t == 0).sum().item()
    lora = n_other = 0
    for n, p in model.named_parameters():  # named_parameters de-duplicates tied weights
        if id(p) in ternary_param_ids:
            continue
        if "lora_" in n:
            lora += p.numel()
        else:
            n_other += p.numel()
    total = n_ternary + n_other
    return {
        "total_params": total,
        "ternary_params": n_ternary,
        "high_precision_params": n_other,
        "ternary_fraction": n_ternary / total if total else 0.0,
        "zero_fraction_of_ternary": n_zero / n_ternary if n_ternary else 0.0,
        "training_only_lora_params": lora,
    }


@torch.no_grad()
def flip_rate(before: dict[str, torch.Tensor], model: nn.Module) -> float:
    """Fraction of ternary codes that changed since `before` (from `snapshot_codes`)."""
    changed = total = 0
    for name, m in model.named_modules():
        if isinstance(m, BitLinear) and name in before:
            t, _ = m.ternary_weight()
            changed += (t != before[name].to(t.device)).sum().item()
            total += t.numel()
    return changed / total if total else 0.0


@torch.no_grad()
def snapshot_codes(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: m.ternary_weight()[0].cpu() for n, m in model.named_modules() if isinstance(m, BitLinear)}


__all__ = ["ternarize_linears", "ternary_report", "flip_rate", "snapshot_codes", "ternarize"]

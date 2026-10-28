"""Ternary linear layers.

BitLinear            training layer: higher-precision latent weight, ternary forward weight (STE).
PackedTernaryLinear  frozen / inference layer: only the packed ternary codes and one scale are stored.

What stays in higher precision (and why)
  * the latent weight in BitLinear during training (needed for small STE updates to accumulate)
  * the per-matrix scale gamma (one fp32 scalar per matrix)
  * activations are 8-bit in the forward pass, but the residual stream, norms,
    softmax, RoPE and the attention scores are bf16/fp32
  * token embeddings, the output head and RMSNorm gains (as in BitNet b1.58)
After export, BitLinear weights exist only as ternary codes + scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import pack, unpack
from .ternary import activation_quant, ste, ternarize, weight_quant


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class BitLinear(nn.Linear):
    """nn.Linear whose forward weight is ternary.

    Args:
        input_norm:  put an RMSNorm inside the layer (original BitNet b1.58 recipe). The 2B4T
                     architecture instead has explicit sub-layer norms, so it uses False.
        act_bits:    activation bits (8 = W1.58A8). None disables activation quantization.
        weight_quant_enabled: False makes the layer an ordinary fp linear (for baselines and
                     for progressive ternarization; see `ternary_mix`).
        lora_rank:   >0 adds a trainable low-rank delta to the frozen latent weight. The delta is
                     added *before* ternarization, so the forward weight stays ternary and the
                     delta is folded away by `merge_lora()`. This is not QLoRA: no dense adapter
                     survives into the exported model.
    """

    def __init__(self, in_features, out_features, bias=False, input_norm=False, act_bits=8,
                 weight_quant_enabled=True, lora_rank=0, lora_alpha=None, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.norm = RMSNorm(in_features) if input_norm else None
        self.act_bits = act_bits
        self.weight_quant_enabled = weight_quant_enabled
        # Progressive ternarization: forward weight = (1-mix)*W + mix*Q(W). mix=1 -> fully ternary.
        self.register_buffer("ternary_mix", torch.tensor(1.0), persistent=False)
        self.lora_rank = lora_rank
        if lora_rank:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_scale = (lora_alpha or lora_rank) / lora_rank
            self.weight.requires_grad_(False)

    def latent_weight(self):
        w = self.weight
        if self.lora_rank:
            w = w + (self.lora_B @ self.lora_A).to(w.dtype) * self.lora_scale
        return w

    def forward_weight(self):
        w = self.latent_weight()
        if not self.weight_quant_enabled:
            return w
        w_q = ste(w, weight_quant(w))
        mix = float(self.ternary_mix)
        return w_q if mix >= 1.0 else (1 - mix) * w + mix * w_q

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        if self.act_bits:
            x = ste(x, activation_quant(x, self.act_bits))
        return F.linear(x, self.forward_weight(), self.bias)

    @torch.no_grad()
    def ternary_weight(self):
        """(int8 codes in {-1,0,1}, fp32 scale) exactly as used in the forward pass."""
        return ternarize(self.latent_weight())

    @torch.no_grad()
    def merge_lora(self):
        if not self.lora_rank:
            return
        self.weight.copy_(self.latent_weight())
        del self.lora_A, self.lora_B
        self.lora_rank = 0
        self.weight.requires_grad_(True)

    def extra_repr(self):
        return (f"{super().extra_repr()}, input_norm={self.norm is not None}, act_bits={self.act_bits}, "
                f"ternary={self.weight_quant_enabled}, lora_rank={self.lora_rank}")


class PackedTernaryLinear(nn.Module):
    """Ternary linear layer holding only packed codes + scale (no latent weight).

    Used for inference and for *frozen* blocks during memory-efficient partial training:
    activations still get gradients (STE through activation quantization), but the
    weights cannot change, so no fp copy, gradient or optimizer state is needed.
    """

    def __init__(self, in_features, out_features, bias=None, input_norm=None, act_bits=8,
                 fmt="2bit", backend="auto"):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.act_bits, self.fmt, self.backend = act_bits, fmt, backend
        n = in_features * out_features
        nbytes = math.ceil(n / (4 if fmt == "2bit" else 5))
        self.register_buffer("packed", torch.zeros(nbytes, dtype=torch.uint8))
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self.norm = input_norm
        self._codes = None  # cached unpacked int8 codes (optional, trades memory for speed)
        self.cache_codes = True

    @classmethod
    def from_bitlinear(cls, layer: BitLinear, fmt="2bit", backend="auto"):
        t, gamma = layer.ternary_weight()
        m = cls(layer.in_features, layer.out_features,
                bias=None if layer.bias is None else layer.bias.detach().clone(),
                input_norm=layer.norm, act_bits=layer.act_bits, fmt=fmt, backend=backend)
        m.packed = pack(t.cpu(), fmt).to(layer.weight.device)
        m.scale = gamma.to(layer.weight.device)
        return m

    def codes(self) -> torch.Tensor:
        if self._codes is not None and self._codes.device == self.packed.device:
            return self._codes
        c = unpack(self.packed, (self.out_features, self.in_features), self.fmt)
        if self.cache_codes:
            self._codes = c
        return c

    def forward(self, x):
        from .kernels import ternary_linear
        if self.norm is not None:
            x = self.norm(x)
        if self.act_bits:
            x = ste(x, activation_quant(x, self.act_bits))
        y = ternary_linear(x, self.codes(), self.scale, backend=self.backend, packed=self.packed)
        return y if self.bias is None else y + self.bias

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, fmt={self.fmt}, act_bits={self.act_bits}"


def ternary_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (BitLinear, PackedTernaryLinear))]


def set_ternary_mix(model: nn.Module, mix: float):
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.ternary_mix.fill_(mix)


@torch.no_grad()
def freeze_to_packed(module: nn.Module, fmt="2bit") -> int:
    """Replace every BitLinear inside `module` with a PackedTernaryLinear. Returns bytes saved."""
    saved = 0
    for name, child in list(module.named_children()):
        if isinstance(child, BitLinear) and not child.lora_rank:
            packed = PackedTernaryLinear.from_bitlinear(child, fmt=fmt)
            saved += child.weight.numel() * child.weight.element_size() - packed.packed.numel()
            setattr(module, name, packed)
        else:
            saved += freeze_to_packed(child, fmt)
    return saved

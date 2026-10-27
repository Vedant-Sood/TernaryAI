"""Ternary weight and 8-bit activation quantizers (BitNet b1.58).

References
  Ma et al., 2024, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764
  Ma et al., 2025, "BitNet b1.58 2B4T Technical Report", arXiv:2504.12285

Weights (absmean, one scale per matrix):
    gamma = mean(|W|)
    T     = clip(round(W / gamma), -1, 1)          T ∈ {-1, 0, +1}
    W_q   = gamma * T

Activations (absmax, one scale per token, int8):
    s   = 127 / max(|x|)
    x_q = clip(round(x * s), -128, 127) / s

During training a higher-precision latent weight W is kept and quantized on
every forward pass. The straight-through estimator (STE) uses the quantized
value in the forward pass and treats the quantizer as the identity in the
backward pass:  W_ste = W + stopgrad(W_q - W).

This is *not* the same thing as 4-bit or 8-bit post-training quantization:
each weight takes exactly one of three values and the scale is shared by the
whole matrix, so a matmul with T needs only additions and subtractions.
"""

from __future__ import annotations

import torch

EPS = 1e-5


def absmean_scale(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean scale gamma = mean|W| (computed in fp32 for stability)."""
    return w.detach().float().abs().mean().clamp(min=EPS)


def ternarize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (T, gamma): int8 codes in {-1, 0, +1} and the fp32 scale."""
    gamma = absmean_scale(w)
    t = (w.detach().float() / gamma).round().clamp(-1, 1).to(torch.int8)
    return t, gamma


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """De-quantized ternary weights, values in {-gamma, 0, +gamma}, same dtype as w."""
    gamma = absmean_scale(w)
    return ((w.float() / gamma).round().clamp(-1, 1) * gamma).to(w.dtype)


def activation_quant(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-token absmax quantization to signed `bits`-bit integers, returned de-quantized."""
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = qmax / xf.abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    return ((xf * scale).round().clamp(-qmax - 1, qmax) / scale).to(x.dtype)


def ste(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward value x_q, gradient of identity w.r.t. x."""
    return x + (x_q - x).detach()


@torch.no_grad()
def ternary_stats(w: torch.Tensor) -> dict:
    """Distribution of ternary codes for one weight matrix."""
    t, gamma = ternarize(w)
    n = t.numel()
    return {
        "numel": n,
        "scale": gamma.item(),
        "frac_neg": (t == -1).sum().item() / n,
        "frac_zero": (t == 0).sum().item() / n,
        "frac_pos": (t == 1).sum().item() / n,
    }

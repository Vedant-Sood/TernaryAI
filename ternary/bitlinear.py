"""BitLinear: the ternary linear layer from BitNet b1.58.

Paper: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
(Ma et al., 2024, arXiv:2402.17764), plus the training recipe notes in
"The Era of 1-bit LLMs: Training Tips, Code and FAQ".

During training we keep a full-precision "latent" weight and quantize it on
every forward pass. Gradients flow through the quantizers with the
straight-through estimator (STE): forward uses the quantized value, backward
pretends the quantizer was the identity.

  weights:     W_q = clip(round(W / mean|W|), -1, 1) * mean|W|     -> {-1, 0, +1} * scale
  activations: x_q = clip(round(x * 127 / max|x|), -128, 127) / (127 / max|x|)  per token

At inference time only the ternary values (2 bits each, or 1.58 bits with
packing) and a single fp scale per matrix need to be stored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization. Returns de-quantized weights (values in {-s, 0, s})."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1) / scale


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token absmax 8-bit quantization. Returns de-quantized activations."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class BitLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with ternary weights and int8 activations.

    Following the reference implementation, the input is RMS-normalized before
    activation quantization so the quantizer sees a well-conditioned range.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x):
        w = self.weight
        x = self.norm(x)
        # STE: value of the quantized tensor, gradient of the identity.
        x_q = x + (activation_quant(x) - x).detach()
        w_q = w + (weight_quant(w) - w).detach()
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def ternary_weight(self):
        """Return (int8 tensor in {-1,0,1}, fp scale) for export / inspection."""
        scale = self.weight.abs().mean().clamp(min=1e-5)
        return (self.weight / scale).round().clamp(-1, 1).to(torch.int8), scale.item()

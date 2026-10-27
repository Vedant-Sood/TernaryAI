"""Compact serialization of ternary codes.

Two lossless formats are supported:

  "2bit"   4 codes per byte (2 bits/weight). Codes: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10.
           Trivial to unpack with shifts and masks, which is what GPU kernels want.
  "base3"  5 codes per byte, since 3**5 = 243 <= 256 (1.6 bits/weight). This is
           close to, but not equal to, the log2(3) = 1.585 bit information bound.
           Unpacking needs integer division, so it is a storage format, not a
           compute format.

Neither format reaches exactly 1.58 bits, and storing weights this way does not
by itself make inference faster: see triforge.quantization.kernels.
"""

from __future__ import annotations

import math

import torch

_POW3 = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32)


def _check_ternary(t: torch.Tensor):
    if t.dtype != torch.int8:
        raise TypeError(f"expected int8 ternary codes, got {t.dtype}")
    if t.numel() and (t.min() < -1 or t.max() > 1):
        raise ValueError("tensor contains values outside {-1, 0, +1}")


def pack_2bit(t: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} tensor (any shape) -> flat uint8 tensor of ceil(n/4) bytes."""
    _check_ternary(t)
    codes = torch.where(t < 0, 2, t).to(torch.uint8).flatten()
    pad = (-codes.numel()) % 4
    if pad:
        codes = torch.cat([codes, codes.new_zeros(pad)])
    c = codes.view(-1, 4)
    return c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)


def unpack_2bit(packed: torch.Tensor, shape) -> torch.Tensor:
    """Inverse of pack_2bit. Returns int8 codes with the given shape."""
    n = math.prod(shape)
    p = packed.to(torch.uint8)
    c = torch.stack([(p >> s) & 0b11 for s in (0, 2, 4, 6)], dim=1).flatten()[:n]
    out = c.to(torch.int8)
    out[out == 2] = -1
    return out.view(*shape)


def pack_base3(t: torch.Tensor) -> torch.Tensor:
    """int8 {-1,0,1} tensor -> flat uint8 tensor of ceil(n/5) bytes (5 trits per byte)."""
    _check_ternary(t)
    digits = (t.flatten().to(torch.int32) + 1)  # {0,1,2}
    pad = (-digits.numel()) % 5
    if pad:
        digits = torch.cat([digits, digits.new_ones(pad)])  # pad with code for 0
    return (digits.view(-1, 5) * _POW3.to(digits.device)).sum(dim=1).to(torch.uint8)


def unpack_base3(packed: torch.Tensor, shape) -> torch.Tensor:
    n = math.prod(shape)
    v = packed.to(torch.int32)
    digits = torch.stack([(v // int(p)) % 3 for p in _POW3], dim=1).flatten()[:n]
    return (digits - 1).to(torch.int8).view(*shape)


PACKERS = {"2bit": (pack_2bit, unpack_2bit), "base3": (pack_base3, unpack_base3)}


def pack(t: torch.Tensor, fmt: str = "2bit") -> torch.Tensor:
    return PACKERS[fmt][0](t)


def unpack(packed: torch.Tensor, shape, fmt: str = "2bit") -> torch.Tensor:
    return PACKERS[fmt][1](packed, shape)


def packed_nbytes(numel: int, fmt: str = "2bit") -> int:
    return math.ceil(numel / (4 if fmt == "2bit" else 5))

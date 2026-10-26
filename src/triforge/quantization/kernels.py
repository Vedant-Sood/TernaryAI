"""Matmul backends for ternary weights.

All backends compute the same thing, y = x @ (gamma * T)^T with T ∈ {-1,0,+1}:

  "reference"  F.linear on T cast to x.dtype, scale applied once afterwards. Always available,
               works on CPU and GPU, and is the correctness oracle for the others.
  "int8"       W1.58A8 integer path: quantize x per token to int8, int8 x int8 -> int32 with
               torch._int_mm, rescale. Exact integer arithmetic; speed depends on the platform.
  "addonly"    multiplication-free: y_j = sum_{i: T_ji=+1} x_i - sum_{i: T_ji=-1} x_i. Written in
               plain PyTorch to demonstrate the arithmetic; it is slow, and is not a fast kernel.
  "triton"     fused 2-bit-unpack + dot product on GPU (only if Triton is installed and the
               tensors are on CUDA, or TRITON_INTERPRET=1 for CPU testing).

Storage savings are not speedups. On a GPU, dense bf16 GEMMs are heavily optimised and a
ternary kernel only wins when the layer is memory-bandwidth bound (small batch decoding) and
the kernel is good enough. Measure with benchmarks/kernel_bench.py before claiming anything.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # optional
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on platform
    HAS_TRITON = False


def ternary_linear_reference(x, codes, scale):
    return F.linear(x, codes.to(x.dtype)) * scale.to(x.dtype)


def ternary_linear_int8(x, codes, scale):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    s = 127.0 / x2.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xq = (x2 * s).round().clamp(-128, 127).to(torch.int8)
    m = xq.shape[0]
    # torch._int_mm needs m > 16 on CUDA and contiguous inputs; pad rows if necessary.
    pad = max(0, 17 - m) if xq.is_cuda else 0
    if pad:
        xq = torch.cat([xq, xq.new_zeros(pad, xq.shape[1])])
    try:
        acc = torch._int_mm(xq, codes.t().contiguous())
    except (RuntimeError, AttributeError):
        acc = (xq.to(torch.int32) @ codes.t().to(torch.int32))
    acc = acc[:m]
    y = acc.float() / s * scale.float()
    return y.to(x.dtype).reshape(*shape[:-1], codes.shape[0])


def ternary_linear_addonly(x, codes, scale):
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    out = x2.new_zeros(x2.shape[0], codes.shape[0])
    pos_r, pos_c = (codes == 1).nonzero(as_tuple=True)
    neg_r, neg_c = (codes == -1).nonzero(as_tuple=True)
    out.index_add_(1, pos_r, x2[:, pos_c])
    out.index_add_(1, neg_r, -x2[:, neg_c])
    return (out * scale.to(x.dtype)).reshape(*shape[:-1], codes.shape[0])


if HAS_TRITON:
    @triton.jit
    def _ternary_2bit_kernel(x_ptr, p_ptr, y_ptr, M, N, K, stride_xm, stride_ym, scale,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # Weight codes are stored row-major (N, K) and packed 4 per byte along the flattened index.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :],
                        mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
            flat = rn[None, :] * K + rk[:, None]               # (BLOCK_K, BLOCK_N)
            byte = tl.load(p_ptr + flat // 4, mask=(rn[None, :] < N) & (rk[:, None] < K), other=0)
            code = (byte.to(tl.int32) >> ((flat % 4) * 2)) & 3
            w = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
            acc += tl.dot(x.to(tl.float32), w.to(tl.float32))
        acc = acc * scale
        tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))

    def ternary_linear_triton(x, packed, scale, out_features):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous().float()
        M, K = x2.shape
        N = out_features
        y = torch.empty(M, N, device=x.device, dtype=torch.float32)
        BM, BN, BK = 16, 32, 32
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _ternary_2bit_kernel[grid](x2, packed, y, M, N, K, x2.stride(0), y.stride(0), float(scale),
                                   BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        return y.to(x.dtype).reshape(*shape[:-1], N)


def triton_available(device: torch.device) -> bool:
    return HAS_TRITON and (device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1")


def ternary_linear(x, codes, scale, backend="auto", packed=None):
    if backend == "auto":
        backend = "reference"
    if backend == "reference":
        return ternary_linear_reference(x, codes, scale)
    if backend == "int8":
        return ternary_linear_int8(x, codes, scale)
    if backend == "addonly":
        return ternary_linear_addonly(x, codes, scale)
    if backend == "triton":
        if packed is None or not triton_available(x.device):
            raise RuntimeError("triton backend needs Triton, CUDA (or TRITON_INTERPRET=1) and packed codes")
        return ternary_linear_triton(x, packed, scale, codes.shape[0])
    raise ValueError(f"unknown backend {backend!r}")

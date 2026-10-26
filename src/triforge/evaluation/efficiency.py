"""Efficiency measurements: size, memory, prefill and decode throughput.

All numbers come from timing real forward passes on the current device. Nothing here is
derived from theoretical bit counts except `theoretical_bits`, which is labelled as such.
"""

from __future__ import annotations

import math
import os
import time

import torch

from ..quantization.convert import ternary_report


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3
    import psutil
    return psutil.Process().memory_info().rss / 1024 ** 3


@torch.no_grad()
def measure_throughput(model, vocab_size, prompt_len=128, gen_tokens=64, batch=1, warmup=1, device=None):
    device = device or next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x = torch.randint(0, vocab_size, (batch, prompt_len), device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    t0 = time.perf_counter()
    model(x)
    _sync(device)
    prefill = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.generate(x, gen_tokens, temperature=0.0)
    _sync(device)
    total = time.perf_counter() - t0
    decode = max(1e-9, total - prefill)
    return {"device": str(device), "batch": batch, "prompt_len": prompt_len, "gen_tokens": gen_tokens,
            "prefill_tokens_per_s": batch * prompt_len / prefill,
            "decode_tokens_per_s": batch * gen_tokens / decode,
            "peak_mem_gib": _peak(device)}


def theoretical_bits(report: dict, fp_bits=16) -> dict:
    """Information-theoretic lower bound (labelled theoretical): log2(3) bits per ternary weight."""
    t, o = report["ternary_params"], report["high_precision_params"]
    return {"theoretical_min_bytes": (t * math.log2(3) + o * fp_bits) / 8,
            "two_bit_packed_bytes": (t * 2 + o * fp_bits) / 8,
            "all_fp16_bytes": (t + o) * 2}


def model_efficiency(model, export_dir=None) -> dict:
    rep = ternary_report(model)
    out = {**rep, **theoretical_bits(rep)}
    if export_dir and os.path.exists(os.path.join(export_dir, "model.safetensors")):
        out["export_model_file_bytes"] = os.path.getsize(os.path.join(export_dir, "model.safetensors"))
    return out

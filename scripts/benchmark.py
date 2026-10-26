"""Efficiency benchmarks: parameters, ternary fraction, disk size, memory and throughput.

    python scripts/benchmark.py --export checkpoints/exports/trimath-cpu-ternary checkpoints/exports/trimath-cpu-fp
    python scripts/benchmark.py --kernels --shape 2560 6912      # matmul backends on one layer shape
    python scripts/benchmark.py --bitnet microsoft/bitnet-b1.58-2B-4T-bf16   # 2B inference footprint (GPU)

Results go to results/benchmarks/*.json. Every number is measured on this machine.
"""
import argparse
import json
import os
import platform
import time

import _common  # noqa: F401
import torch
import torch.nn.functional as F

from triforge.evaluation.efficiency import measure_throughput, model_efficiency
from triforge.quantization import activation_quant, kernels, pack
from triforge.training.config import resolve_path


def bench_kernels(n_in, n_out, tokens, device, reps):
    t = torch.randint(-1, 2, (n_out, n_in), dtype=torch.int8, device=device)
    g = torch.tensor(0.02, device=device)
    dt = torch.bfloat16 if device.type == "cuda" else torch.float32
    x = activation_quant(torch.randn(tokens, n_in, device=device)).to(dt)
    dense = torch.randn(n_out, n_in, device=device, dtype=dt)
    packed = pack(t.cpu()).to(device)
    cases = {"dense_fp_matmul": lambda: F.linear(x, dense),
             "ternary_reference": lambda: kernels.ternary_linear(x, t, g, "reference"),
             "ternary_int8": lambda: kernels.ternary_linear(x, t, g, "int8")}
    if n_in * n_out <= 4_000_000:
        cases["ternary_addonly"] = lambda: kernels.ternary_linear(x, t, g, "addonly")
    if kernels.triton_available(device):
        cases["ternary_triton_2bit"] = lambda: kernels.ternary_linear(x.float(), t, g, "triton", packed=packed)
    out = {}
    for name, fn in cases.items():
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        out[name] = {"ms": (time.perf_counter() - t0) / reps * 1e3}
    base = out["dense_fp_matmul"]["ms"]
    for v in out.values():
        v["relative_to_dense"] = round(v["ms"] / base, 3)
    return {"shape": [n_out, n_in], "tokens": tokens, "device": str(device), "dtype": str(dt), "reps": reps,
            "weight_bytes": {"dense_" + str(dt).replace("torch.", ""): n_in * n_out * dense.element_size(),
                             "int8_codes": n_in * n_out, "packed_2bit": packed.numel()}, "results": out}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export", nargs="*", default=[])
    p.add_argument("--bitnet", default=None)
    p.add_argument("--kernels", action="store_true")
    p.add_argument("--shape", nargs=2, type=int, default=[2560, 6912])
    p.add_argument("--tokens", nargs="*", type=int, default=[1, 16, 512])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = resolve_path("results/benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    host = {"platform": platform.platform(), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if args.kernels:
        res = {"host": host, "cases": [bench_kernels(args.shape[0], args.shape[1], n, device, args.reps)
                                       for n in args.tokens]}
        path = os.path.join(out_dir, f"kernels_{args.shape[1]}x{args.shape[0]}_{device.type}.json")
        json.dump(res, open(path, "w"), indent=2)
        print(json.dumps(res, indent=2))
    for e in args.export:
        from triforge.inference.export import load_export
        path = resolve_path(e)
        m = load_export(path, device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        res = {"host": host, "export": e, "status": m.export_meta.get("status"), **model_efficiency(m, path),
               "throughput": measure_throughput(m, m.cfg.vocab_size, min(args.prompt_len, m.cfg.max_seq_len // 2),
                                                args.gen_tokens, device=device)}
        name = os.path.basename(path.rstrip("/"))
        json.dump(res, open(os.path.join(out_dir, f"{name}_{device.type}.json"), "w"), indent=2)
        print(json.dumps(res, indent=2))
    if args.bitnet:
        from triforge.huggingface.bitnet_weights import load_bitnet
        from triforge.quantization import freeze_to_packed
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        m = load_bitnet(args.bitnet, dtype=torch.bfloat16, gradient_checkpointing=False, max_seq_len=2048)
        for blk in m.blocks:
            freeze_to_packed(blk)  # inference: ternary codes only (int8 cache + 2-bit packed)
        m = m.to(device).eval()
        res = {"host": host, "model": args.bitnet, **model_efficiency(m),
               "throughput": measure_throughput(m, m.cfg.vocab_size, args.prompt_len, args.gen_tokens, device=device)}
        json.dump(res, open(os.path.join(out_dir, f"bitnet2b_{device.type}.json"), "w"), indent=2)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

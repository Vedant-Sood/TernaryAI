"""Detect hardware and pick a safe runtime configuration for a run config, or explain why it cannot fit.

    python scripts/profile_hardware.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
    python scripts/profile_hardware.py --config ... --assume-vram 8 --assume-ram 32   # plan for another machine
"""
import argparse
import json

import _common  # noqa: F401

from triforge.architectures import TriForgeConfig
from triforge.hardware import DoesNotFit, choose_runtime, detect, estimate_inference_memory, estimate_training_memory
from triforge.training import load_config

BITNET_2B4T = dict(vocab_size=128256, max_seq_len=4096, d_model=2560, n_layers=30, n_heads=20, n_kv_heads=5,
                   d_ff=6912, mlp="relu2_glu", subln=True, bitlinear_input_norm=False, tie_embeddings=True,
                   name="bitnet-b1.58-2B-4T")


def arch_of(cfg):
    m = cfg["model"]
    if m.get("source") == "scratch":
        return TriForgeConfig.from_dict(m["arch"])
    if m.get("source") == "bitnet":
        return TriForgeConfig(**BITNET_2B4T)
    raise SystemExit("profile_hardware supports model.source scratch|bitnet")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--assume-vram", type=float, default=None, help="GiB, plan for a GPU that is not attached")
    p.add_argument("--assume-ram", type=float, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    info = detect()
    if args.assume_vram:
        info.cuda_available, info.gpu_total_gib = True, args.assume_vram
        info.gpu_free_gib = args.assume_vram - 0.6  # display + driver reserve on a laptop
    if args.assume_ram:
        info.ram_total_gib = info.ram_available_gib = args.assume_ram
    arch = arch_of(cfg)
    m = cfg["model"]
    strategy = {"lora_rank": m.get("lora_rank", 0) if m.get("strategy") == "lora" else 0,
                "train_embeddings": m.get("train_embeddings", m.get("source") == "scratch"),
                "frozen_format": "packed" if not m.get("frozen_cache_codes", True) else "int8"}
    tl = m.get("trainable_layers")
    if m.get("strategy") == "partial" and isinstance(tl, str) and tl.startswith("last:"):
        strategy["n_trainable_layers"] = int(tl.split(":")[1])
    elif m.get("strategy") == "partial" and tl:
        strategy["n_trainable_layers"] = len(tl)
    r = cfg["runtime"]
    requested = {"seq_len": r.get("seq_len"), "micro_batch_size": r.get("micro_batch_size"),
                 "gradient_checkpointing": m.get("gradient_checkpointing", True), "optimizer": cfg["optim"]["name"]}
    counts = arch.param_counts()
    out = {"hardware": info.__dict__, "model": {"name": arch.name, **counts}, "strategy": strategy,
           "requested": requested}
    est = estimate_training_memory(counts, arch.d_model, arch.d_ff, arch.n_layers, arch.vocab_size,
                                   requested["micro_batch_size"], requested["seq_len"], optimizer=requested["optimizer"],
                                   grad_checkpointing=requested["gradient_checkpointing"],
                                   n_trainable_layers=strategy.get("n_trainable_layers"),
                                   train_embeddings=strategy["train_embeddings"], lora_rank=strategy["lora_rank"],
                                   frozen_format=strategy["frozen_format"], chunked_loss=bool(r.get("loss_chunk")))
    out["estimate_for_requested_gib"] = est.as_gib()
    out["inference_gib"] = {f: estimate_inference_memory(counts, arch.d_model, arch.n_layers, arch.n_kv_heads,
                                                         arch.head_dim, 2048, weight_format=f)
                            for f in ("bf16", "int8-cache", "packed")}
    try:
        out["recommended_runtime"] = choose_runtime(arch, info, strategy, {**requested, "micro_batch_size": "auto"})
    except DoesNotFit as e:
        out["recommended_runtime"] = None
        out["does_not_fit"] = str(e)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()

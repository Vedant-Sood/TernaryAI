"""Compare trained runs: quality (val loss / perplexity) and memory footprint.

Storage model:
  fp baseline     every weight stored as fp16 (2 bytes)
  ternary model   BitLinear weights packed at 2 bits each (4 per byte) + one fp16
                  scale per matrix; embeddings, head and norms stay fp16
  (the 1.58-bit information-theoretic bound, 5 trits per byte, is shown too)

Usage: python compare.py runs/ternary runs/fp
"""

import json
import math
import os
import sys

import torch

from ternary import GPT, BitLinear, ModelConfig


def pack_2bit(t: torch.Tensor) -> bytes:
    """Pack int8 values in {-1,0,1} into 2 bits each (00=0, 01=+1, 10=-1)."""
    codes = (t.flatten() % 3).to(torch.uint8)  # -1 -> 2, 0 -> 0, 1 -> 1
    pad = (-codes.numel()) % 4
    codes = torch.cat([codes, torch.zeros(pad, dtype=torch.uint8)]).view(-1, 4)
    packed = codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)
    return packed.numpy().tobytes()


def footprint(model: GPT):
    ternary_bytes_2bit = ternary_bytes_158 = 0.0
    n_ternary = zeros = 0
    ternary_ids = set()
    for mod in model.modules():
        if isinstance(mod, BitLinear):
            q, _ = mod.ternary_weight()
            ternary_ids.add(id(mod.weight))
            n_ternary += q.numel()
            zeros += (q == 0).sum().item()
            ternary_bytes_2bit += len(pack_2bit(q)) + 2
            ternary_bytes_158 += math.ceil(q.numel() / 5) + 2
    fp_params = sum(p.numel() for p in model.parameters() if id(p) not in ternary_ids)
    fp_bytes = fp_params * 2
    return {
        "fp16_bytes": fp_bytes + n_ternary * 2,  # same model if everything were fp16
        "packed_2bit_bytes": fp_bytes + ternary_bytes_2bit,
        "packed_1.58bit_bytes": fp_bytes + ternary_bytes_158,
        "ternary_zero_frac": zeros / n_ternary if n_ternary else None,
    }


def main(run_dirs):
    rows = []
    for d in run_dirs:
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        ckpt = torch.load(os.path.join(d, "ckpt.pt"), map_location="cpu")
        model = GPT(ModelConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        fp = footprint(model)
        is_ternary = ckpt["cfg"]["linear"] == "ternary"
        rows.append({
            "run": m["name"],
            "params": m["params"]["total"],
            "val_loss": m["final_val_loss"],
            "val_ppl": math.exp(m["final_val_loss"]),
            "train_time_s": m["train_time_s"],
            "weights_kb": (fp["packed_2bit_bytes"] if is_ternary else fp["fp16_bytes"]) / 1024,
            "weights_kb_1.58": (fp["packed_1.58bit_bytes"] if is_ternary else fp["fp16_bytes"]) / 1024,
            "zero_frac": fp["ternary_zero_frac"],
        })

    hdr = f"{'run':<12}{'params':>10}{'val loss':>10}{'val ppl':>9}{'train s':>9}{'weights KB':>12}{'@1.58b KB':>11}{'zeros':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        z = f"{r['zero_frac']:.1%}" if r["zero_frac"] is not None else "-"
        print(f"{r['run']:<12}{r['params']:>10,}{r['val_loss']:>10.4f}{r['val_ppl']:>9.3f}"
              f"{r['train_time_s']:>9.0f}{r['weights_kb']:>12.0f}{r['weights_kb_1.58']:>11.0f}{z:>8}")
    with open(os.path.join("runs", "comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/ternary", "runs/fp"])

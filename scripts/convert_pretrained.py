"""Phase B: post-training ternarization of a dense pretrained model, measured before any training.

    python scripts/convert_pretrained.py --model HuggingFaceTB/SmolLM2-360M --data data/phase_b/v1-fineweb

Reports validation loss / perplexity of (a) the dense model, (b) the same model with every decoder
linear ternarized (absmean, int8 activations), and (c) weights-only ternarization (fp activations).
Writes results/phase_b/<model>_ptq.json. This is conversion only; it is NOT training.
"""
import argparse
import json
import math
import os

import _common  # noqa: F401
import torch

from triforge.datasets.loader import TokenDataset
from triforge.huggingface.hf_models import load_dense, ternarize_hf
from triforge.quantization import ternary_report
from triforge.training.config import resolve_path


@torch.no_grad()
def val(model, ds, seq, iters, device, bs):
    g = torch.Generator().manual_seed(0)
    ls = []
    for _ in range(iters):
        x, y, m = ds.batch(bs, seq, device, generator=g)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            ls.append(model(x, y, loss_mask=m)[1].item())
    v = sum(ls) / len(ls)
    return {"val_loss": v, "val_ppl": math.exp(min(v, 50))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    ds = TokenDataset(resolve_path(args.data), "val")
    res = {"model": args.model, "seq_len": args.seq_len, "batches": args.iters, "batch": args.batch}
    dense = load_dense(args.model, device=device).eval()
    res["dense"] = val(dense, ds, args.seq_len, args.iters, device, args.batch)
    del dense
    for name, act_bits in (("ternary_w_fp_act", None), ("ternary_w1.58a8", 8)):
        m = load_dense(args.model, device=device).eval()
        ternarize_hf(m, act_bits=act_bits)
        res[name] = {**val(m, ds, args.seq_len, args.iters, device, args.batch), **ternary_report(m)}
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out = resolve_path(f"results/phase_b/{args.model.replace('/', '__')}_ptq.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate an exported model (or a HF model) on a specialist's benchmark suite.

    python scripts/evaluate.py --export checkpoints/exports/trimath-cpu-ternary --specialist trimath \
        --data data/trimath/v0-cpu --benchmarks synthetic gsm8k --limit 200
    python scripts/evaluate.py --hf microsoft/bitnet-b1.58-2B-4T-bf16 --template bitnet --specialist trimath ...

Writes results/<name>/<benchmark>.json (summary) and .jsonl (per-example outputs). Use --limit for
quick runs; the summary always records n, so partial runs are visibly partial.
"""
import argparse
import json
import math
import os
import time

import _common  # noqa: F401
import torch

from triforge.datasets.loader import TokenDataset
from triforge.evaluation import benchmarks
from triforge.evaluation.runner import SUITES, eval_code, eval_exact_match, eval_multiple_choice
from triforge.training.config import resolve_path


def load(args, device):
    if args.export:
        from triforge.inference.engine import load_tokenizer_from_export
        from triforge.inference.export import load_export
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        m = load_export(resolve_path(args.export), device=device, dtype=dtype)
        return m, load_tokenizer_from_export(resolve_path(args.export)), m.export_meta.get("template", "triforge"), \
            m.export_meta
    if args.bitnet:  # the unmodified BitNet 2B4T, loaded into TriForge's implementation
        from triforge.datasets.tokenizer import load_tokenizer
        from triforge.huggingface.bitnet_weights import load_bitnet
        m = load_bitnet(args.bitnet, device=device, gradient_checkpointing=False,
                        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
        return m, load_tokenizer({"type": "hf", "name": args.bitnet}), "bitnet", {"status": "base model (no TriForge training)"}
    from triforge.datasets.tokenizer import load_tokenizer
    from triforge.huggingface.hf_models import load_dense
    m = load_dense(args.hf, device=device).eval()
    return m, load_tokenizer({"type": "hf", "name": args.hf}), args.template, {"status": "dense baseline"}


@torch.no_grad()
def val_loss(model, data_dir, seq_len, iters, device):
    ds = TokenDataset(resolve_path(data_dir), "val")
    g = torch.Generator().manual_seed(0)
    losses = []
    for _ in range(iters):
        x, y, m = ds.batch(8, seq_len, device, generator=g)
        losses.append(model(x, y, loss_mask=m)[1].item())
    v = sum(losses) / len(losses)
    return {"metric": "val_loss", "val_loss": v, "val_ppl": math.exp(v), "n_batches": iters, "seq_len": seq_len}


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--export")
    src.add_argument("--hf")
    src.add_argument("--bitnet")
    p.add_argument("--template", default="triforge")
    p.add_argument("--specialist", required=True, choices=list(SUITES))
    p.add_argument("--benchmarks", nargs="*", default=None, help="default: the specialist's suite + val loss")
    p.add_argument("--data", default=None, help="dataset dir (for val loss and the synthetic test set)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    model, tok, template, meta = load(args, device)
    name = args.name or os.path.basename((args.export or args.hf or args.bitnet).rstrip("/"))
    out_dir = resolve_path(os.path.join("results", name))
    os.makedirs(out_dir, exist_ok=True)
    benches = args.benchmarks or (["val_loss"] + SUITES[args.specialist])
    summary = {"model": name, "status": meta.get("status"), "device": str(device), "template": template,
               "decoding": "greedy", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "results": {}}
    for b in benches:
        t0 = time.time()
        jl = os.path.join(out_dir, f"{b}.jsonl")
        try:
            if b == "val_loss":
                res = val_loss(model, args.data, min(256, model.cfg.max_seq_len), 20, device)
            elif b == "synthetic":
                rows = benchmarks.synthetic_test(args.data)[: args.limit]
                code = [r for r in rows if r.get("tests")]
                plain = [r for r in rows if not r.get("tests")]
                res = {"exact_match": eval_exact_match(model, tok, plain, template, "prompt",
                                                       args.max_new_tokens or 96, device, jl)}
                if code:
                    res["code_pass@1"] = eval_code(model, tok, code, template, "synthetic",
                                                   args.max_new_tokens or 160, device, jl.replace(".jsonl", "_code.jsonl"))
                by_task = {}
                for line in open(jl):
                    r = json.loads(line)
                    task = next(x["task"] for x in plain if x["id"] == r["id"])
                    d = by_task.setdefault(task, [0, 0])
                    d[0] += r["correct"]
                    d[1] += 1
                res["by_task"] = {k: {"accuracy": c / n, "n": n} for k, (c, n) in sorted(by_task.items())}
            elif b in ("humaneval", "mbpp"):
                rows = benchmarks.LOADERS[b]()[: args.limit]
                res = eval_code(model, tok, rows, template, b, args.max_new_tokens or 384, device, jl)
            elif b in ("gsm8k", "math500"):
                rows = benchmarks.LOADERS[b]()[: args.limit]
                res = eval_exact_match(model, tok, rows, template, "question", args.max_new_tokens or 512, device, jl)
            elif b == "arc_challenge":
                rows = benchmarks.arc()[: args.limit]
                res = eval_multiple_choice(model, tok, rows, template, device, jl)
            else:
                raise ValueError(f"unknown benchmark {b}")
            res["limit"] = args.limit
        except Exception as e:  # record, never invent a number
            res = {"error": f"{type(e).__name__}: {e}", "status": "not executed"}
        res["seconds"] = round(time.time() - t0, 1)
        summary["results"][b] = res
        with open(os.path.join(out_dir, f"{b}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"[eval] {name} {b}: {json.dumps(res)[:300]}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

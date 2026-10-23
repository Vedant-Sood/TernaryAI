"""Train a small GPT with ternary (BitNet b1.58) or full-precision linear layers.

Examples:
  python train.py --linear ternary
  python train.py --linear fp
  python train.py --linear ternary --n_layer 6 --n_embd 384 --block_size 256 --max_iters 5000   # GPU-sized

Writes runs/<name>/{ckpt.pt,metrics.json,sample.txt}.
"""

import argparse
import json
import math
import os
import time

import torch

from data import CharDataset
from ternary import GPT, ModelConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--linear", choices=["ternary", "fp"], default="ternary")
    p.add_argument("--name", default=None, help="run name (default: the --linear value)")
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=None,
                   help="peak LR (default: 3e-3 ternary, 1e-3 fp; BitNet needs a larger LR)")
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(it, args, peak):
    if it < args.warmup:
        return peak * (it + 1) / args.warmup
    progress = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    min_lr = peak * args.min_lr_frac
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, ds, args, gen):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(args.eval_iters):
            x, y = ds.batch(split, args.batch_size, args.block_size, gen)
            _, loss = model(x.to(args.device), y.to(args.device))
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    args = get_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    peak_lr = args.lr or (3e-3 if args.linear == "ternary" else 1e-3)
    name = args.name or args.linear
    out_dir = os.path.join("runs", name)
    os.makedirs(out_dir, exist_ok=True)

    ds = CharDataset()
    cfg = ModelConfig(vocab_size=ds.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                      n_head=args.n_head, n_embd=args.n_embd, linear=args.linear)
    model = GPT(cfg).to(args.device)
    params = model.param_breakdown()
    print(f"[{name}] params: {params}")

    # No weight decay on norms / embeddings, as usual.
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=peak_lr, betas=(0.9, 0.95))

    batch_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(0)
    history = []
    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = evaluate(model, ds, args, eval_gen)
            elapsed = time.time() - t0
            history.append({"iter": it, "train": losses["train"], "val": losses["val"], "time_s": elapsed})
            print(f"[{name}] iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")
        if it == args.max_iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it, args, peak_lr)
        x, y = ds.batch("train", args.batch_size, args.block_size, batch_gen)
        _, loss = model(x.to(args.device), y.to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    train_time = time.time() - t0
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict()}, os.path.join(out_dir, "ckpt.pt"))

    model.eval()
    start = torch.tensor([ds.encode("ROMEO:\n")], device=args.device)
    sample = ds.decode(model.generate(start, 400, temperature=0.8, top_k=40)[0].tolist())
    with open(os.path.join(out_dir, "sample.txt"), "w") as f:
        f.write(sample)
    print(f"[{name}] sample:\n{sample}\n")

    metrics = {"name": name, "args": vars(args), "peak_lr": peak_lr, "params": params,
               "final_val_loss": history[-1]["val"], "final_train_loss": history[-1]["train"],
               "train_time_s": train_time, "history": history}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

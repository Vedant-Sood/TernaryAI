"""Export the newest (or a given) checkpoint of a run to the packed ternary format.

    python scripts/export.py --run checkpoints/trimath/cpu_ternary --out checkpoints/exports/trimath-cpu
    python scripts/export.py --run ... --hf-bf16 out_dir   # BitNet models: also write HF BitNet layout
"""
import argparse
import json
import os

import _common  # noqa: F401
import torch

from triforge.datasets.tokenizer import load_tokenizer
from triforge.inference.export import export_model, export_size_report
from triforge.training import build_model, latest_checkpoint, load_checkpoint, load_config
from triforge.training.config import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run output directory (contains config.yaml)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--status", default="trained", choices=["trained", "untrained", "converted-not-finetuned"])
    p.add_argument("--packing", default="2bit", choices=["2bit", "base3"])
    p.add_argument("--hf-bf16", default=None, help="also export HF BitNet bf16 layout (BitNet runs only)")
    args = p.parse_args()
    run = resolve_path(args.run)
    cfg = load_config(os.path.join(run, "config.yaml"))
    model = build_model(cfg["model"])
    ck = args.checkpoint or latest_checkpoint(run)
    if ck is None and args.status == "trained":
        raise SystemExit(f"no checkpoint in {run}; refusing to export an untrained model as 'trained'")
    state = load_checkpoint(ck, model) if ck else {}
    tok = load_tokenizer(cfg["data"]["tokenizer"])
    fp_dtype = torch.bfloat16 if cfg["model"].get("source") == "bitnet" else torch.float16
    prov = {"run": args.run, "checkpoint": ck, "step": state.get("step"), "tokens_seen": state.get("tokens_seen"),
            "config": cfg.get("run", {}).get("config_path"), "data_dir": cfg["data"].get("dir"),
            "base_model": cfg["model"].get("path")}
    info = export_model(model, resolve_path(args.out), status=args.status, specialist=cfg["run"]["specialist"],
                        tokenizer=tok, template=cfg["data"].get("template", "triforge"), fmt=args.packing,
                        fp_dtype=fp_dtype, provenance=prov)
    if args.hf_bf16:
        from triforge.huggingface.bitnet_weights import resolve_local, save_hf_bf16
        save_hf_bf16(model, resolve_path(args.hf_bf16), resolve_local(cfg["model"]["path"]))
    print(json.dumps({**info, **export_size_report(resolve_path(args.out))}, indent=2))


if __name__ == "__main__":
    main()

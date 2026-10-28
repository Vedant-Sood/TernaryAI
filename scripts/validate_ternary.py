"""Check that every forward weight of a model is genuinely ternary, and report code distributions.

    python scripts/validate_ternary.py --export checkpoints/exports/trimath-cpu-ternary
    python scripts/validate_ternary.py --config configs/trimath/smoke.yaml      # freshly initialised model
"""
import argparse
import json

import _common  # noqa: F401
import torch

from triforge.quantization import BitLinear, PackedTernaryLinear, ternary_report
from triforge.training import build_model, load_config


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--export")
    g.add_argument("--config")
    args = p.parse_args()
    if args.export:
        from triforge.inference.export import load_export
        model = load_export(args.export)
    else:
        model = build_model(load_config(args.config)["model"])
    layers, bad = {}, []
    for name, m in model.named_modules():
        if isinstance(m, BitLinear):
            w = m.forward_weight().detach()
            t, gamma = m.ternary_weight()
        elif isinstance(m, PackedTernaryLinear):
            t, gamma = m.codes(), m.scale
            w = t.float() * gamma
        else:
            continue
        vals = torch.unique(t).tolist()
        ok = set(vals) <= {-1, 0, 1} and torch.allclose(w.float(), t.float() * gamma, atol=1e-6)
        if not ok:
            bad.append(name)
        n = t.numel()
        layers[name] = {"shape": list(t.shape), "scale": float(gamma), "neg": (t == -1).sum().item() / n,
                        "zero": (t == 0).sum().item() / n, "pos": (t == 1).sum().item() / n, "ternary": ok}
    out = {"summary": ternary_report(model), "all_forward_weights_ternary": not bad, "violations": bad,
           "layers": layers}
    print(json.dumps(out, indent=2))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

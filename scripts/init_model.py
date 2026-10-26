"""Initialise the model described by a config and print its parameter / memory accounting.

    python scripts/init_model.py --config configs/tricode/cpu_validation.yaml [--save out_dir]
"""
import argparse
import json

import _common  # noqa: F401

from triforge.quantization import ternary_report
from triforge.training import build_model, load_config
from triforge.training.build import trainable_summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--save", default=None, help="write an *untrained* export here (status=untrained)")
    a = p.parse_args()
    cfg = load_config(a.config)
    model = build_model(cfg["model"])
    print(json.dumps({**trainable_summary(model), **ternary_report(model)}, indent=2))
    if a.save:
        from triforge.datasets.tokenizer import load_tokenizer
        from triforge.inference.export import export_model
        print(export_model(model, a.save, status="untrained", specialist=cfg["run"]["specialist"],
                           tokenizer=load_tokenizer(cfg["data"]["tokenizer"])))

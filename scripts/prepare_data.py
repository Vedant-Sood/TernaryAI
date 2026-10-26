"""Build tokenized, decontaminated dataset versions for one or more specialist configs.

    python scripts/prepare_data.py --config configs/tricode/cpu_validation.yaml configs/trimath/cpu_validation.yaml ...

When several configs share a BPE tokenizer path that does not exist yet, the tokenizer is
trained once on the union of their training data, so all specialists share one vocabulary.
"""
import argparse
import os

import _common  # noqa: F401

from triforge.datasets import pipeline
from triforge.datasets.tokenizer import train_bpe
from triforge.evaluation.benchmarks import decontamination_texts
from triforge.training import load_config
from triforge.training.config import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    cfgs = [load_config(c, args.set) for c in args.config]
    for c in cfgs:
        c["data"]["tokenizer"] = dict(c["data"]["tokenizer"])
        if c["data"]["tokenizer"]["type"] == "bpe":
            c["data"]["tokenizer"]["path"] = resolve_path(c["data"]["tokenizer"]["path"])
    # shared tokenizer
    bpe = {c["data"]["tokenizer"]["path"] for c in cfgs if c["data"]["tokenizer"]["type"] == "bpe"}
    for path in bpe:
        if os.path.exists(path):
            continue
        users = [c for c in cfgs if c["data"]["tokenizer"].get("path") == path]
        corpus = []
        for c in users:
            for spec in c["data"]["sources"]:
                for ex in pipeline.iter_source(c["run"]["specialist"], spec):
                    corpus.append(ex.get("text") or (ex["prompt"] + "\n" + ex["response"]))
        vs = users[0]["data"]["tokenizer"]["vocab_size"]
        print(f"[tokenizer] training shared BPE ({vs}) on {len(corpus)} documents from "
              f"{[c['run']['specialist'] for c in users]}")
        train_bpe(corpus, vs, path)
    bench_cache = {}
    for c in cfgs:
        names = tuple(c["data"].get("decontaminate", {}).get("benchmarks", []))
        if names not in bench_cache:
            bench_cache[names] = decontamination_texts(names) if c["data"].get("decontaminate", {}).get("enabled", True) else {}
        out_root = resolve_path("data")
        c["data"]["version"] = c["data"].get("version", "v0")
        manifest = pipeline.build(c, out_root=out_root, benchmark_texts=bench_cache[names])
        print(f"[done] {c['run']['specialist']} {manifest['version']}: {manifest['splits']}")


if __name__ == "__main__":
    main()

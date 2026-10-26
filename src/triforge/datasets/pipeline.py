"""Build a tokenized, decontaminated, versioned dataset for one specialist.

    python scripts/prepare_data.py --config configs/trimath/cpu_validation.yaml

Output (under data/<specialist>/<version>/):
    train.tokens.npy / train.mask.npy     packed token stream + loss mask (1 = trained on)
    val.tokens.npy   / val.mask.npy
    test.jsonl                            held-out examples kept as text, for task evaluation
    manifest.json                         sources, licences, revisions, filter counts, mixture, hashes
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
import sysconfig
import time
from collections import Counter

import numpy as np

from . import quality, synthetic
from .registry import SOURCES
from .tokenizer import format_example, load_tokenizer, train_bpe


# --------------------------------------------------------------------------- source loaders

def _iter_hf(src, spec):
    from datasets import load_dataset
    ds = load_dataset(src.hf_id, spec.get("config", src.config), split=spec.get("split", src.split),
                      streaming=True, revision=spec.get("revision"))
    f = src.fields
    for row in ds:
        if "text" in f:
            yield {"text": row[f["text"]], "prompt": "", "response": row[f["text"]]}
            continue
        ex = {"prompt": row.get(f.get("prompt", "prompt"), ""), "response": row.get(f.get("response", "response"), "")}
        if "answer" in f:
            ex["answer"] = str(row.get(f["answer"], ""))
        yield ex


def _iter_stdlib(spec):
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    files = [p for p in files if "/test" not in p and "site-packages" not in p]
    random.Random(spec.get("seed", 0)).shuffle(files)
    for p in files[: spec.get("max_files", 500)]:
        try:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        if 200 < len(text) < spec.get("max_chars", 40_000):
            yield {"text": f"# file: {os.path.relpath(p, root)}\n{text}", "prompt": "", "response": text}


def _iter_jsonl(spec):
    with open(spec["path"]) as fh:
        for line in fh:
            yield json.loads(line)


def iter_source(specialist, spec):
    key = spec["key"]
    if key == "synthetic":
        train, _ = synthetic.generate_split(specialist, spec.get("n", 10000), 0, seed=spec.get("seed", 0))
        yield from train
    elif key == "python_stdlib":
        yield from _iter_stdlib(spec)
    elif key == "jsonl":
        yield from _iter_jsonl(spec)
    else:
        src = SOURCES[key]
        if not src.train_ok:
            raise ValueError(f"source {key} is marked train_ok=False ({src.notes or src.license})")
        yield from _iter_hf(src, spec)


# --------------------------------------------------------------------------- build

def build(cfg: dict, out_root="data", benchmark_texts: dict | None = None, log=print) -> dict:
    run, data = cfg["run"], cfg["data"]
    specialist = run["specialist"]
    version = data.get("version", "v0")
    out = os.path.join(out_root, specialist, version)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rng = random.Random(data.get("seed", 0))

    decon = None
    if data.get("decontaminate", {}).get("enabled", True) and benchmark_texts:
        decon = quality.Decontaminator(n=data.get("decontaminate", {}).get("ngram", 13))
        for name, texts in benchmark_texts.items():
            decon.add_benchmark(name, texts)

    lsh = quality.MinHashLSH() if data.get("near_dedup", True) else None
    seen = set()
    stats = Counter()
    per_source = {}
    splits = {"train": [], "val": [], "test": []}

    for spec in data["sources"]:
        key = spec["key"]
        n_kept = 0
        cap = spec.get("max_examples")
        keep_re = re.compile(spec["keep_regex"], re.I) if spec.get("keep_regex") else None
        drop_re = re.compile(spec["drop_regex"], re.I) if spec.get("drop_regex") else None
        for ex in iter_source(specialist, spec):
            stats["seen"] += 1
            if keep_re and not keep_re.search(ex.get("prompt", "")):
                stats[f"drop:{key} outside domain (keep_regex)"] += 1
                continue
            if drop_re and drop_re.search(ex.get("prompt", "")):
                stats[f"drop:{key} drop_regex"] += 1
                continue
            if "text" not in ex:
                reason = quality.is_malformed(ex)
                if reason:
                    stats[f"drop:{reason}"] += 1
                    continue
                if spec.get("require_valid_python") and not quality.python_blocks_parse(ex["response"]):
                    stats["drop:python syntax error"] += 1
                    continue
            k = quality.exact_key(ex)
            if k in seen:
                stats["drop:exact duplicate"] += 1
                continue
            seen.add(k)
            text = ex.get("text") or (ex["prompt"] + "\n" + ex["response"])
            # Templated synthetic items share most shingles by construction, so near-dedup is
            # only applied to natural data (exact dedup still applies to everything).
            near = spec.get("near_dedup", key != "synthetic")
            if lsh is not None and near and len(text) < 20_000 and \
                    lsh.query_insert(k, quality.shingles(ex["prompt"] or text[:2000])):
                stats["drop:near duplicate"] += 1
                continue
            if decon is not None and decon.contaminated(ex.get("prompt") or text):
                stats["drop:benchmark contamination"] += 1
                continue
            ex["source"] = key
            split = quality.split_of(ex, data.get("val_frac", 0.02), data.get("test_frac", 0.02))
            for _ in range(max(1, int(spec.get("weight", 1)))):
                splits[split].append(ex)
                if split != "train":
                    break
            n_kept += 1
            if cap and n_kept >= cap:
                break
        per_source[key] = {"kept": n_kept, "weight": spec.get("weight", 1),
                           "license": SOURCES[key].license if key in SOURCES else spec.get("license", "unknown"),
                           "hf_id": SOURCES[key].hf_id if key in SOURCES else None,
                           "revision": spec.get("revision", "main (unpinned)")}
        log(f"[data] {specialist}/{key}: kept {n_kept}")

    # task-specific held-out set from disjoint synthetic seeds (never in train)
    synth_test = data.get("synthetic_test", 0)
    if synth_test:
        _, test = synthetic.generate_split(specialist, 0, synth_test, seed=data.get("seed", 0))
        train_prompts = {e.get("prompt") for e in splits["train"]}
        uniq = {}
        for e in test:
            if e["prompt"] not in train_prompts:
                uniq.setdefault(e["prompt"], e)
        test = list(uniq.values())
        for e in test:
            e["source"] = "synthetic_test"
        splits["test"].extend(test)

    tok_spec = data["tokenizer"]
    if tok_spec["type"] == "bpe" and not os.path.exists(tok_spec["path"]):
        log(f"[data] training BPE tokenizer ({tok_spec['vocab_size']} tokens) -> {tok_spec['path']}")
        corpus = [e.get("text") or (e["prompt"] + "\n" + e["response"]) for e in splits["train"]]
        if tok_spec.get("extra_corpus"):  # e.g. other specialists, to share one tokenizer
            for p in tok_spec["extra_corpus"]:
                corpus.extend(json.loads(line)["text"] for line in open(p))
        train_bpe(corpus, tok_spec["vocab_size"], tok_spec["path"])
    tok = load_tokenizer(tok_spec)

    shard_info = {}
    template = data.get("template", "triforge")
    for split in ("train", "val"):
        exs = splits[split]
        rng.shuffle(exs)
        ids, mask = [], []
        for ex in exs:
            a, m = format_example(tok, ex, template)
            ids.extend(a)
            mask.extend(m)
        dtype = np.uint16 if tok.vocab_size < 65535 else np.uint32
        arr, marr = np.asarray(ids, dtype=dtype), np.asarray(mask, dtype=np.uint8)
        np.save(os.path.join(out, f"{split}.tokens.npy"), arr)
        np.save(os.path.join(out, f"{split}.mask.npy"), marr)
        shard_info[split] = {"examples": len(exs), "tokens": int(arr.size), "trained_tokens": int(marr.sum()),
                             "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    with open(os.path.join(out, "test.jsonl"), "w") as fh:
        for ex in splits["test"]:
            fh.write(json.dumps(ex) + "\n")
    shard_info["test"] = {"examples": len(splits["test"])}

    manifest = {"specialist": specialist, "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tokenizer": {**tok_spec, "vocab_size": tok.vocab_size}, "template": template,
                "sources": per_source, "filters": dict(stats), "splits": shard_info,
                "decontamination": {"ngram": decon.n, "benchmarks": decon.sources} if decon else None,
                "build_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    log(f"[data] wrote {out}: {json.dumps(shard_info)}")
    return manifest

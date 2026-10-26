"""Benchmark loaders. Everything is cached under data/benchmarks/ after the first download.

Official sources are used where they exist (HumanEval and MBPP from their GitHub releases,
GSM8K from openai/grade-school-math); MATH-500 and ARC come from the Hugging Face Hub.
Each loader returns a list of dicts with a stable "id".
"""

from __future__ import annotations

import gzip
import io
import json
import os
import urllib.request

from ..training.config import resolve_path

CACHE = "data/benchmarks"


def _fetch(url: str, name: str) -> bytes:
    path = resolve_path(os.path.join(CACHE, name))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    with open(path, "rb") as f:
        return f.read()


def humaneval():
    raw = _fetch("https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
                 "HumanEval.jsonl.gz")
    rows = [json.loads(line) for line in gzip.open(io.BytesIO(raw), "rt")]
    return [{"id": r["task_id"], **r} for r in rows]


def mbpp(split="test"):
    raw = _fetch("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
                 "mbpp.jsonl")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    lo, hi = {"test": (11, 510), "validation": (511, 600), "train": (601, 974), "prompt": (1, 10)}[split]
    return [{"id": f"mbpp/{r['task_id']}", **r} for r in rows if lo <= r["task_id"] <= hi]


def gsm8k(split="test"):
    raw = _fetch(f"https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl",
                 f"gsm8k_{split}.jsonl")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return [{"id": f"gsm8k/{i}", "question": r["question"], "solution": r["answer"],
             "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(rows)]


def _hf_rows(repo, config=None, split="test"):
    from datasets import load_dataset
    return list(load_dataset(repo, config, split=split))


def math500():
    return [{"id": f"math500/{i}", "question": r["problem"], "answer": r["answer"], "subject": r.get("subject")}
            for i, r in enumerate(_hf_rows("HuggingFaceH4/MATH-500"))]


def arc(config="ARC-Challenge", split="test"):
    out = []
    for r in _hf_rows("allenai/ai2_arc", config, split):
        out.append({"id": r["id"], "question": r["question"], "choices": r["choices"]["text"],
                    "labels": r["choices"]["label"], "answer": r["answerKey"]})
    return out


def synthetic_test(data_dir: str):
    with open(os.path.join(resolve_path(data_dir), "test.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    return [{"id": f"synthetic/{i}", **r} for i, r in enumerate(rows) if r.get("source") == "synthetic_test"]


LOADERS = {"humaneval": humaneval, "mbpp": mbpp, "gsm8k": gsm8k, "math500": math500, "arc_challenge": arc}


def decontamination_texts(names) -> dict:
    """Benchmark prompts used to build the n-gram blocklist for training data."""
    texts = {}
    for n in names:
        try:
            rows = LOADERS[n]()
        except Exception as e:  # offline or blocked: report, don't silently pretend
            texts[n] = []
            print(f"[decontam] WARNING: could not load {n} ({type(e).__name__}); not decontaminated against it")
            continue
        texts[n] = [r.get("prompt") or r.get("question") or r.get("text", "") for r in rows]
    return texts

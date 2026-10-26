"""Task evaluation: generation-based (exact match / pass@1) and log-likelihood multiple choice.

All results record the number of examples evaluated, the decoding settings and per-example
outputs, so partial runs (--limit) are visibly partial.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

from ..datasets.tokenizer import prompt_ids
from .answers import answers_match, extract_answer, extract_code
from .code_exec import run_code


@torch.no_grad()
def generate_text(model, tok, prompt, template, max_new_tokens=256, temperature=0.0, device="cpu", raw=False):
    ids = tok.encode(prompt) if raw else prompt_ids(tok, prompt, template)
    max_new_tokens = min(max_new_tokens, model.cfg.max_seq_len // 2)
    max_ctx = model.cfg.max_seq_len - max_new_tokens
    ids = ids[-max_ctx:]
    x = torch.tensor([ids], device=device)
    out = model.generate(x, max_new_tokens, temperature=temperature, eos_id=tok.eos_id)
    return tok.decode(out[0, len(ids):].tolist())


def _record(path, rows):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def eval_exact_match(model, tok, examples, template, key_q="prompt", max_new_tokens=128, device="cpu", out=None):
    rows, correct = [], 0
    t0 = time.time()
    for ex in examples:
        text = generate_text(model, tok, ex[key_q], template, max_new_tokens, device=device)
        pred = extract_answer(text)
        ok = answers_match(pred, str(ex["answer"]))
        correct += ok
        rows.append({"id": ex["id"], "pred": pred, "gold": ex["answer"], "correct": ok, "output": text[:2000]})
    _record(out, rows)
    n = len(examples)
    return {"metric": "exact_match", "accuracy": correct / n if n else 0.0, "n": n, "correct": correct,
            "seconds": time.time() - t0}


def eval_code(model, tok, examples, template, kind, max_new_tokens=384, device="cpu", out=None, timeout=10):
    rows, passed = [], 0
    t0 = time.time()
    for ex in examples:
        if kind == "humaneval":
            prompt = f"Complete this Python function.\n```python\n{ex['prompt']}```"
            text = generate_text(model, tok, prompt, template, max_new_tokens, device=device)
            code = extract_code(text)
            if f"def {ex['entry_point']}" not in code:
                code = ex["prompt"] + code
            program = code + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})\n"
        elif kind == "mbpp":
            prompt = ex["text"] + "\nYour code should pass these tests:\n" + "\n".join(ex["test_list"])
            text = generate_text(model, tok, prompt, template, max_new_tokens, device=device)
            code = extract_code(text)
            program = ex.get("test_setup_code", "") + "\n" + code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
        else:  # synthetic "code/implement" with asserts
            text = generate_text(model, tok, ex["prompt"], template, max_new_tokens, device=device)
            code = extract_code(text)
            program = code + "\n" + "\n".join(ex["tests"]) + "\n"
        res = run_code(program, timeout=timeout)
        passed += res["passed"]
        rows.append({"id": ex["id"], "passed": res["passed"], "output": text[:2000], "stderr": res["stderr"][-500:]})
    _record(out, rows)
    n = len(examples)
    return {"metric": "pass@1 (greedy)", "accuracy": passed / n if n else 0.0, "n": n, "correct": passed,
            "seconds": time.time() - t0}


@torch.no_grad()
def eval_multiple_choice(model, tok, examples, template, device="cpu", out=None):
    """Pick the choice with the highest mean token log-likelihood given the question."""
    rows, correct = [], 0
    t0 = time.time()
    for ex in examples:
        q = ex["question"] + "\nAnswer:"
        p_ids = prompt_ids(tok, q, template)
        scores = []
        for choice in ex["choices"]:
            c_ids = tok.encode(" " + choice)
            ids = (p_ids + c_ids)[-model.cfg.max_seq_len:]
            x = torch.tensor([ids], device=device)
            logits, _ = model(x)
            lp = F.log_softmax(logits[0, :-1].float(), -1)
            tgt = x[0, 1:]
            tok_lp = lp.gather(-1, tgt[:, None])[:, 0][-len(c_ids):]
            scores.append(tok_lp.mean().item())
        pred = ex["labels"][int(torch.tensor(scores).argmax())]
        ok = pred == ex["answer"]
        correct += ok
        rows.append({"id": ex["id"], "pred": pred, "gold": ex["answer"], "correct": ok, "scores": scores})
    _record(out, rows)
    n = len(examples)
    return {"metric": "mc_loglik_acc", "accuracy": correct / n if n else 0.0, "n": n, "correct": correct,
            "seconds": time.time() - t0}


SUITES = {
    "tricode": ["synthetic", "humaneval", "mbpp"],
    "trimath": ["synthetic", "gsm8k", "math500"],
    "trireason": ["synthetic", "arc_challenge"],
}

import json

import numpy as np
import pytest

from triforge.datasets import pipeline, quality, synthetic
from triforge.datasets.loader import TokenDataset
from triforge.datasets.tokenizer import format_example, load_tokenizer, train_bpe
from triforge.evaluation.answers import answers_match


@pytest.mark.parametrize("spec", ["tricode", "trimath", "trireason"])
def test_synthetic_is_deterministic_and_disjoint(spec):
    a = synthetic.generate(spec, 50, seed=3)
    b = synthetic.generate(spec, 50, seed=3)
    assert a == b
    train, test = synthetic.generate_split(spec, 300, 100)
    assert not {e["prompt"] for e in train} & {e["prompt"] for e in test}


@pytest.mark.parametrize("spec", ["trimath", "trireason", "tricode"])
def test_synthetic_responses_end_with_their_answer(spec):
    from triforge.evaluation.answers import extract_answer
    for ex in synthetic.generate(spec, 60, seed=1):
        if ex["task"] == "code/implement":
            assert synthetic.run_python_tests(ex["answer"], ex["tests"])
        else:
            assert answers_match(extract_answer(ex["response"]), ex["answer"]), ex


def test_math_answers_are_correct():
    for ex in synthetic.generate("trimath", 200, seed=7):
        if ex["task"] == "math/linear_equation":
            a, rest = ex["prompt"].split("Solve for x: ")[1].split("x + (")
            b, c = rest.split(") = ")
            assert int(a) * int(ex["answer"]) + int(b) == int(c)


def test_quality_filters():
    assert quality.is_malformed({"prompt": "", "response": "x"})
    assert quality.is_malformed({"prompt": "hello there", "response": "bad �"})
    assert quality.is_malformed({"prompt": "What is two plus two?", "response": "4"}) is None
    assert quality.python_blocks_parse("```python\nx = 1\n```")
    assert not quality.python_blocks_parse("```python\ndef f(:\n```")


def test_near_duplicate_detection():
    lsh = quality.MinHashLSH()
    base = "the quick brown fox jumps over the lazy dog near the river bank today " * 3
    assert not lsh.query_insert("a", quality.shingles(base))
    assert lsh.query_insert("b", quality.shingles(base + " again"))
    assert not lsh.query_insert("c", quality.shingles("completely different text about matrices and eigenvalues "
                                                      "of symmetric operators on hilbert spaces"))


def test_decontamination():
    d = quality.Decontaminator(n=8)
    d.add_benchmark("bench", ["Natalia sold clips to 48 of her friends in April, and then she sold half as many"])
    assert d.contaminated("Q: natalia sold clips to 48 of her friends in april, and then something")
    assert not d.contaminated("A totally unrelated question about prime numbers and their gaps")


def test_hash_split_is_stable():
    ex = {"prompt": "p", "response": "r"}
    assert quality.split_of(ex, 0.1, 0.1) == quality.split_of(dict(ex), 0.1, 0.1)


def test_pipeline_end_to_end(tmp_path):
    tokpath = str(tmp_path / "tok.json")
    corpus = [e["prompt"] + "\n" + e["response"] for e in synthetic.generate("trimath", 2000, 0)]
    train_bpe(corpus, 512, tokpath)
    cfg = {"run": {"specialist": "trimath"},
           "data": {"version": "t", "sources": [{"key": "synthetic", "n": 1500, "seed": 0}],
                    "tokenizer": {"type": "bpe", "path": tokpath, "vocab_size": 512}, "template": "triforge",
                    "val_frac": 0.05, "test_frac": 0.0, "synthetic_test": 50}}
    bench = {"gsm8k": ["Compute ((5 + 7) * 3) - 4 would be contaminated if this exact long string were present"]}
    man = pipeline.build(cfg, out_root=str(tmp_path / "data"), benchmark_texts=bench, log=lambda *a: None)
    d = tmp_path / "data" / "trimath" / "t"
    assert man["splits"]["train"]["tokens"] > 1000 and man["splits"]["test"]["examples"] > 0
    assert man["sources"]["synthetic"]["license"].startswith("generated")
    tokens = np.load(d / "train.tokens.npy")
    mask = np.load(d / "train.mask.npy")
    assert tokens.shape == mask.shape and 0 < mask.mean() < 1  # prompts masked out, responses trained
    test = [json.loads(line) for line in open(d / "test.jsonl")]
    tok = load_tokenizer(cfg["data"]["tokenizer"])
    ds = TokenDataset(str(d), "train", seed=0)
    x, y, m = ds.batch(4, 32)
    assert x.shape == (4, 32) and torch_equal_shift(ds, x, y)
    ids, msk = format_example(tok, {"prompt": "Compute 1 + 1.", "response": "Answer: 2"})
    assert len(ids) == len(msk) and msk[0] == 0 and msk[-1] == 1 and ids[-1] == tok.eos_id


def torch_equal_shift(ds, x, y):
    return bool((x[:, 1:] == y[:, :-1]).all())


def test_loader_state_resumes_identically(tmp_path):
    np.save(tmp_path / "train.tokens.npy", np.arange(1000, dtype=np.uint16))
    a = TokenDataset(str(tmp_path), "train", seed=5)
    a.batch(2, 8)
    st = a.state()
    nxt = a.batch(2, 8)[0]
    b = TokenDataset(str(tmp_path), "train", seed=99)
    b.set_state(st)
    assert (b.batch(2, 8)[0] == nxt).all()

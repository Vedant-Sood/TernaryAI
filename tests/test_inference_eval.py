import pytest

from triforge.architectures import TriForgeConfig, TriForgeLM
from triforge.evaluation.answers import answers_match, extract_answer, extract_boxed, extract_code
from triforge.evaluation.code_exec import run_code
from triforge.hardware import DoesNotFit, choose_runtime, detect, estimate_training_memory
from triforge.inference.router import route, score


@pytest.mark.parametrize("prompt,expected", [
    ("Write a Python function to reverse a linked list", "tricode"),
    ("Why does my JavaScript code throw a TypeError: undefined is not a function?", "tricode"),
    ("Prove that there are infinitely many primes", "trimath"),
    ("Compute the integral of x^2 from 0 to 3", "trimath"),
    ("A 2 kg block slides with acceleration 3 m/s^2. What force acts on it?", "trireason"),
    ("All birds are animals. Some animals fly. Is the conclusion valid logic?", "trireason"),
])
def test_router(prompt, expected):
    assert route(prompt).specialist == expected, score(prompt)


def test_router_override_and_default():
    assert route("Prove it", override="tricode").specialist == "tricode"
    assert route("hello there").reason.startswith("no domain cues")
    with pytest.raises(ValueError):
        route("x", override="gpt")


def test_answer_extraction():
    assert extract_answer("work...\nAnswer: 42") == "42"
    assert extract_answer("so #### 1,234") == "1,234"
    assert extract_boxed("thus \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
    assert answers_match("1,234", "1234") and answers_match("0.5", "1/2") and answers_match("\\frac{1}{2}", "0.5")
    assert not answers_match("3", "4") and not answers_match(None, "1")
    assert extract_code("text\n```python\nx=1\n```") == "x=1\n"


def test_code_exec_pass_fail_timeout():
    assert run_code("assert 1 + 1 == 2")["passed"]
    assert not run_code("assert 1 + 1 == 3")["passed"]
    r = run_code("while True: pass", timeout=1)
    assert not r["passed"] and r["stderr"] == "timeout"


def test_memory_estimator_and_profiler():
    cfg = TriForgeConfig(vocab_size=128256, max_seq_len=4096, d_model=2560, n_layers=30, n_heads=20, n_kv_heads=5,
                         d_ff=6912, mlp="relu2_glu", subln=True, bitlinear_input_norm=False, tie_embeddings=True)
    c = cfg.param_counts()
    assert 2.3e9 < c["total"] < 2.5e9 and 2.0e9 < c["ternary"] < 2.1e9
    full = estimate_training_memory(c, 2560, 6912, 30, 128256, 1, 2048, optimizer="adamw")
    part = estimate_training_memory(c, 2560, 6912, 30, 128256, 1, 2048, optimizer="adamw8bit",
                                    n_trainable_layers=8, train_embeddings=False)
    assert full.total > 20 * 2 ** 30 > part.total  # full fine-tuning cannot fit 8 GB; partial is far smaller
    info = detect()
    info.cuda_available, info.gpu_free_gib = True, 7.4
    with pytest.raises(DoesNotFit, match="NOT shrunk"):
        choose_runtime(cfg, info, {}, {"optimizer": "adamw", "seq_len": 2048, "gradient_checkpointing": True})
    r = choose_runtime(cfg, info, {"n_trainable_layers": 8, "train_embeddings": False},
                       {"seq_len": 2048, "gradient_checkpointing": True})
    assert r["estimate_gib"]["total_gpu"] <= 7.4


def test_engine_and_api(tmp_path):
    from fastapi.testclient import TestClient

    from triforge.datasets.tokenizer import train_bpe
    from triforge.inference.api import create_app
    from triforge.inference.engine import Engine
    from triforge.inference.export import export_model
    tok = train_bpe(["def f(x): return x", "integral of x", "force and energy"] * 30, 300, str(tmp_path / "t.json"))
    m = TriForgeLM(TriForgeConfig(vocab_size=tok.vocab_size, max_seq_len=64, d_model=32, n_layers=1, n_heads=2))
    export_model(m, str(tmp_path / "exp"), status="untrained", specialist="tricode", tokenizer=tok)
    (tmp_path / "models.yaml").write_text(f"models:\n  tricode: {{specialist: tricode, path: {tmp_path / 'exp'}}}\n"
                                          f"  trimath: {{specialist: trimath, path: {tmp_path / 'missing'}}}\n")
    eng = Engine(str(tmp_path / "models.yaml"), device="cpu")
    client = TestClient(create_app(eng))
    assert client.get("/health").json()["ok"]
    av = client.get("/models").json()
    assert av["tricode"]["installed"] and not av["trimath"]["installed"]
    r = client.post("/generate", json={"prompt": "write a python function", "max_new_tokens": 5, "temperature": 0})
    body = r.json()
    assert r.status_code == 200 and body["stats"]["model"] == "tricode" and body["stats"]["status"] == "untrained"
    assert body["stats"]["router"]["specialist"] == "tricode" and body["stats"]["new_tokens"] >= 1
    assert client.post("/generate", json={"prompt": "x", "model": "trimath"}).status_code == 404
    assert client.post("/unload", json={"model": "tricode"}).status_code == 200
    assert client.post("/route", json={"prompt": "solve the equation 2x + 3 = 7"}).json()["specialist"] == "trimath"


def test_eval_runner_on_tiny_model(tmp_path):
    from triforge.datasets import synthetic
    from triforge.datasets.tokenizer import train_bpe
    from triforge.evaluation.runner import eval_code, eval_exact_match, eval_multiple_choice
    exs = synthetic.generate("trimath", 200, 0)
    tok = train_bpe([e["prompt"] + e["response"] for e in exs], 400, str(tmp_path / "t.json"))
    m = TriForgeLM(TriForgeConfig(vocab_size=tok.vocab_size, max_seq_len=128, d_model=32, n_layers=1, n_heads=2))
    rows = [{"id": str(i), **e} for i, e in enumerate(exs[:3])]
    r = eval_exact_match(m, tok, rows, "triforge", max_new_tokens=8, out=str(tmp_path / "o.jsonl"))
    assert r["n"] == 3 and 0 <= r["accuracy"] <= 1
    code = [{"id": "c", "prompt": "Write add", "tests": ["assert add(1, 2) == 3"]}]
    assert eval_code(m, tok, code, "triforge", "synthetic", max_new_tokens=4)["n"] == 1
    mc = [{"id": "q", "question": "2+2?", "choices": ["4", "5"], "labels": ["A", "B"], "answer": "A"}]
    assert eval_multiple_choice(m, tok, mc, "triforge")["n"] == 1


def test_model_card_renders_only_measured_results(tmp_path):
    import json

    from triforge.huggingface.model_card import render
    s = {"model": "trimath-x", "status": "trained", "results": {
        "val_loss": {"val_loss": 0.5, "val_ppl": 1.65, "n_batches": 20},
        "gsm8k": {"metric": "exact_match", "accuracy": 0.1, "n": 100},
        "math500": {"error": "ConnectionError: blocked", "status": "not executed"}}}
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(s))
    md = render("trimath", "status text", [str(p)])
    assert md.startswith("---\nlicense: mit") and "| trimath-x | gsm8k (exact_match) | 10.0% | 100 | trained |" in md
    assert "not executed" in md and "status text" in md
    assert "_No evaluation results yet._" in render("tricode", "s", [])

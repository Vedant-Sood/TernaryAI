"""Checkpoint integrity, resume determinism, export round trip in a fresh process."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from triforge.architectures import TriForgeConfig, TriForgeLM
from triforge.inference.export import export_model, load_export
from triforge.training import Trainer, load_checkpoint, save_checkpoint
from triforge.training.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(tmp_path, steps, out="run"):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        np.save(d / f"{split}.tokens.npy", rng.integers(0, 64, 5000).astype(np.uint16))
        np.save(d / f"{split}.mask.npy", np.ones(5000, dtype=np.uint8))
    return {
        "run": {"name": "t", "specialist": "trimath", "output_dir": str(tmp_path / out), "seed": 7},
        "model": {"source": "scratch", "arch": dict(vocab_size=64, max_seq_len=32, d_model=32, n_layers=2,
                                                    n_heads=2, d_ff=64)},
        "data": {"dir": str(d), "tokenizer": {"type": "bpe", "path": "unused"}},
        "optim": {"name": "adamw", "lr": 3e-3, "max_steps": steps, "warmup_steps": 2},
        "runtime": {"device": "cpu", "micro_batch_size": 4, "seq_len": 16, "grad_accum_steps": 2,
                    "log_interval": 1, "checkpoint_interval": 3, "keep_last": 5},
        "eval": {"interval": 100, "val_iters": 2},
    }


def _losses(path):
    return [r["loss"] for r in map(json.loads, open(path)) if r.get("event") == "train"]


def test_resume_reproduces_uninterrupted_run(tmp_path):
    full = Trainer(_cfg(tmp_path, 6, "full"), log=lambda *a: None)
    full.fit()
    a = Trainer(_cfg(tmp_path, 3, "split"), log=lambda *a: None)
    a.fit()
    b = Trainer(_cfg(tmp_path, 6, "split"), log=lambda *a: None)
    b.resume("auto")
    assert b.step == 3
    b.fit()
    lf = _losses(tmp_path / "full" / "metrics.jsonl")
    ls = _losses(tmp_path / "split" / "metrics.jsonl")
    assert len(lf) == 6 and len(ls) == 6
    assert np.allclose(lf, ls, atol=1e-5), (lf, ls)


def test_time_cap_stops_early_and_saves(tmp_path):
    cfg = _cfg(tmp_path, 1000, "capped")
    cfg["runtime"]["max_train_minutes"] = 0
    t = Trainer(cfg, log=lambda *a: None)
    t.fit()
    assert t.step == 0
    final = [r for r in map(json.loads, open(tmp_path / "capped" / "metrics.jsonl")) if r.get("final")]
    assert final and final[0]["stopped_on_time"] is True


def test_checkpoint_integrity_check(tmp_path):
    m = TriForgeLM(TriForgeConfig(vocab_size=50, max_seq_len=16, d_model=32, n_layers=1, n_heads=2))
    p = save_checkpoint(str(tmp_path), 1, m)
    load_checkpoint(p, m)
    with open(os.path.join(p, "model.safetensors"), "r+b") as f:
        f.seek(-8, 2)
        f.write(b"\x00" * 8)
    with pytest.raises(RuntimeError, match="integrity"):
        load_checkpoint(p, m)


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"optim": {"lr": 1.0, "name": "adamw"}, "runtime": {"a": 1}}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"inherit": ["base.yaml"], "optim": {"lr": 2.0}}))
    c = load_config(tmp_path / "child.yaml", ["runtime.a=5", "optim.betas=[0.9, 0.99]"])
    assert c["optim"] == {"lr": 2.0, "name": "adamw", "betas": [0.9, 0.99]} and c["runtime"]["a"] == 5


@pytest.mark.parametrize("name", ["tricode", "trimath", "trireason"])
def test_all_specialist_configs_load(name):
    for f in os.listdir(os.path.join(ROOT, "configs", name)):
        c = load_config(os.path.join(ROOT, "configs", name, f))
        assert c["run"]["specialist"] == name
        assert {"model", "data", "optim", "runtime"} <= set(c)
        if c["model"].get("source") == "scratch":
            TriForgeConfig.from_dict(c["model"]["arch"])


def test_export_reload_fresh_process(tmp_path):
    torch.manual_seed(0)
    from triforge.datasets.tokenizer import train_bpe
    tok = train_bpe(["hello world " * 50, "compute 1 + 2 answer 3"] * 20, 300, str(tmp_path / "t.json"))
    cfg = TriForgeConfig(vocab_size=tok.vocab_size, max_seq_len=64, d_model=32, n_layers=2, n_heads=2)
    m = TriForgeLM(cfg).eval()
    x = torch.randint(0, tok.vocab_size, (1, 12))
    ref = m(x)[0]
    info = export_model(m, str(tmp_path / "exp"), status="untrained", specialist="trimath", tokenizer=tok)
    assert info["ternary_modules"] == 14
    back = load_export(str(tmp_path / "exp"))
    assert torch.allclose(back(x)[0], ref, atol=1e-2)  # embeddings/norms stored as fp16
    code = (f"import sys, torch; sys.path.insert(0, {os.path.join(ROOT, 'src')!r})\n"
            "from triforge.inference.export import load_export\n"
            "from triforge.inference.engine import load_tokenizer_from_export\n"
            "from triforge.datasets.tokenizer import prompt_ids\n"
            f"m = load_export({str(tmp_path / 'exp')!r}); t = load_tokenizer_from_export({str(tmp_path / 'exp')!r})\n"
            "ids = prompt_ids(t, 'compute 1 + 2'); out = m.generate(torch.tensor([ids]), 8, temperature=0)\n"
            "assert out.shape[1] >= len(ids) + 1; print(m.export_meta['status'])\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "untrained"


def test_export_refuses_unknown_status(tmp_path):
    m = TriForgeLM(TriForgeConfig(vocab_size=20, max_seq_len=8, d_model=16, n_layers=1, n_heads=2))
    with pytest.raises(ValueError):
        export_model(m, str(tmp_path), status="great", specialist="trimath")


def test_trainable_export_reload_is_exact(tmp_path):
    m = TriForgeLM(TriForgeConfig(vocab_size=40, max_seq_len=16, d_model=32, n_layers=1, n_heads=2)).eval()
    export_model(m, str(tmp_path), status="untrained", specialist="trimath", fp_dtype=torch.float32)
    t = load_export(str(tmp_path), trainable=True).eval()
    x = torch.randint(0, 40, (1, 8))
    assert torch.allclose(m(x)[0], t(x)[0], atol=1e-5)


def test_dense_to_ternary_conversion_and_distillation(tmp_path):
    cfg = _cfg(tmp_path, 3, "fp")
    cfg["model"]["arch"].update(linear="fp", bitlinear_input_norm=False, act_bits=None)
    Trainer(cfg, log=lambda *a: None).fit()
    from triforge.quantization import BitLinear
    from triforge.training.build import build_model
    m = build_model({"source": "convert_triforge", "path": str(tmp_path / "fp")})
    assert sum(isinstance(x, BitLinear) for x in m.modules()) == 14
    c2 = _cfg(tmp_path, 2, "qat")
    c2["model"] = {"source": "convert_triforge", "path": str(tmp_path / "fp")}
    c2["distill"] = {"enabled": True, "alpha": 0.5, "temperature": 2.0,
                     "teacher": {"source": "triforge_run", "path": str(tmp_path / "fp")}}
    c2["optim"]["progressive_ternarization_steps"] = 2
    ev = Trainer(c2, log=lambda *a: None).fit()
    assert ev["val_loss"] > 0

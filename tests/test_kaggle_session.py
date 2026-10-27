"""Kaggle session chaining: find the previous session's checkpoint under /kaggle/input and restore it."""

import importlib.util
import os
import sys
import time


from triforge.architectures import TriForgeConfig, TriForgeLM
from triforge.training import load_config, save_checkpoint
from triforge.training.checkpoint import latest_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location("kaggle_session", os.path.join(ROOT, "scripts", "kaggle_session.py"))
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)

REL = "checkpoints/trimath/bitnet2b_partial_kaggle"


def _model():
    return TriForgeLM(TriForgeConfig(vocab_size=20, max_seq_len=8, d_model=16, n_layers=1, n_heads=2))


def test_restores_newest_checkpoint_from_inputs(tmp_path):
    m = _model()
    old = tmp_path / "input" / "session-1" / "TernaryAI" / REL
    new = tmp_path / "input" / "session-2" / "TernaryAI" / REL
    save_checkpoint(str(old), 50, m)
    save_checkpoint(str(new), 50, m)
    save_checkpoint(str(new), 100, m)
    (new / "metrics.jsonl").write_text('{"event": "train", "step": 100}\n')
    src, step = ks.find_previous_run(str(tmp_path / "input"), REL)
    assert step == 100 and src == str(new)
    dst = tmp_path / "work" / REL
    ck = ks.restore_run(src, str(dst), log=lambda *a: None)
    assert latest_checkpoint(str(dst)) == ck and ks.ckpt_step(ck) == 100
    assert (dst / "metrics.jsonl").exists()
    # a local checkpoint that is already as new is kept
    assert ks.restore_run(str(old), str(dst), log=lambda *a: None) == ck


def test_no_input_means_fresh_start(tmp_path):
    assert ks.find_previous_run(str(tmp_path / "missing"), REL) == (None, -1)


def test_restore_data(tmp_path):
    src = tmp_path / "input" / "s" / "TernaryAI" / "data" / "trimath" / "v1-bitnet"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text("{}")
    (src / "train.tokens.npy").write_bytes(b"x")
    dst = tmp_path / "work" / "data" / "trimath" / "v1-bitnet"
    assert ks.restore_data(str(tmp_path / "input"), "data/trimath/v1-bitnet", str(dst), log=lambda *a: None)
    assert (dst / "train.tokens.npy").exists()
    assert not ks.restore_data(str(tmp_path / "none"), "data/x", str(tmp_path / "work" / "x"), log=lambda *a: None)


def test_minutes_left():
    assert abs(ks.minutes_left(time.time() - 3600, 11.5, 20) - (690 - 60 - 20)) < 1


def test_kaggle_configs_use_fp16_and_fp32_weights():
    for s in ("tricode", "trimath", "trireason"):
        for preset in ("partial", "quick"):
            c = load_config(os.path.join(ROOT, "configs", s, f"bitnet2b_{preset}_kaggle_t4.yaml"))
            base = load_config(os.path.join(ROOT, "configs", s, f"bitnet2b_{preset}_rtx4060.yaml"))
            assert c["runtime"]["precision"] == "fp16" and c["runtime"]["param_dtype"] == "fp32"
            assert c["run"]["output_dir"] == base["run"]["output_dir"] + "_kaggle"
            assert c["data"] == base["data"] and c["optim"] == base["optim"]

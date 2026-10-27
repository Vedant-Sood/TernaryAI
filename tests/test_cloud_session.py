"""Cloud session chaining: restore the previous session's checkpoint and mirror new ones to Drive."""

import importlib.util
import os
import sys
import time


from triforge.architectures import TriForgeConfig, TriForgeLM
from triforge.training import load_config, save_checkpoint
from triforge.training.checkpoint import latest_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
_spec = importlib.util.spec_from_file_location("cloud_session", os.path.join(ROOT, "scripts", "cloud_session.py"))
ks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ks)

REL = "checkpoints/trimath/bitnet2b_partial_t4"


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
    ck = ks.copy_run(src, str(dst), log=lambda *a: None)
    assert latest_checkpoint(str(dst)) == ck and ks.ckpt_step(ck) == 100
    assert (dst / "metrics.jsonl").exists()
    # a local checkpoint that is already as new is kept
    assert ks.copy_run(str(old), str(dst), log=lambda *a: None) == ck


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


def test_cloud_presets_change_only_the_runtime():
    for s in ("tricode", "trimath", "trireason"):
        for preset in ("partial", "quick"):
            base = load_config(os.path.join(ROOT, "configs", s, f"bitnet2b_{preset}_rtx4060.yaml"))
            for profile, prec in (("t4", "fp16"), ("l4", "bf16")):
                c = load_config(os.path.join(ROOT, "configs", s, f"bitnet2b_{preset}_{profile}.yaml"))
                assert c["runtime"]["precision"] == prec and c["runtime"]["param_dtype"] == "fp32"
                assert c["run"]["output_dir"] == base["run"]["output_dir"] + "_" + profile
                assert c["data"] == base["data"] and c["optim"] == base["optim"]
                rt, rb = c["runtime"], base["runtime"]  # same effective batch
                assert rt["micro_batch_size"] * rt["grad_accum_steps"] == rb["micro_batch_size"] * rb["grad_accum_steps"]


def test_sync_keeps_one_checkpoint_and_truncates_the_old_one(tmp_path):
    m = _model()
    work, drive = tmp_path / "work" / REL, tmp_path / "drive" / REL
    save_checkpoint(str(work), 50, m, keep_last=1)
    ks.copy_run(str(work), str(drive), log=lambda *a: None, prune=True)
    old = drive / "ckpt" / "step_0000050" / "model.safetensors"
    assert old.stat().st_size > 0
    save_checkpoint(str(work), 100, m, keep_last=1)
    ck = ks.copy_run(str(work), str(drive), log=lambda *a: None, prune=True)
    assert ks.ckpt_step(ck) == 100 and latest_checkpoint(str(drive)) == ck
    assert sorted(os.listdir(drive / "ckpt")) == ["latest", "step_0000100"]
    # nothing to copy: nothing changes
    assert ks.copy_run(str(tmp_path / "nothing"), str(drive), log=lambda *a: None) is None

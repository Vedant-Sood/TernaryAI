"""Run one Kaggle session of a 2B TriForge fine-tune, resuming where the previous session stopped.

    python scripts/kaggle_session.py --specialist trimath --preset partial

Kaggle kills a notebook after 12 hours and only keeps what is in /kaggle/working. A long run is
therefore a chain of sessions. Each session:

  1. restores the newest checkpoint (and the tokenized data) found under /kaggle/input, i.e. the
     output of the previous session added to the notebook as an input,
  2. prepares the data if no copy was found,
  3. trains with `--resume auto` and a wall-clock cap computed from the time left in the session,
     so the run evaluates and saves before Kaggle stops it,
  4. optionally exports the model once the schedule is finished.

Exit code: 0 when the session ended normally (finished or out of time), else the trainer's code.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import _common  # noqa: F401

from triforge.training import load_config
from triforge.training.checkpoint import latest_checkpoint
from triforge.training.config import resolve_path

ROOT = _common.ROOT


def ckpt_step(ckpt_dir):
    with open(os.path.join(ckpt_dir, "state.json")) as f:
        return json.load(f)["step"]


def find_previous_run(input_root, rel_out):
    """Newest complete checkpoint of this run anywhere under input_root, as (run_dir, step)."""
    best = (None, -1)
    for latest in glob.glob(os.path.join(input_root, "**", rel_out, "ckpt", "latest"), recursive=True):
        run_dir = os.path.dirname(os.path.dirname(latest))
        ck = latest_checkpoint(run_dir)
        if ck is not None and ckpt_step(ck) > best[1]:
            best = (run_dir, ckpt_step(ck))
    return best


def restore_run(src_run, dst_run, log=print):
    """Copy the newest checkpoint and the metrics log of src_run into dst_run (if it is newer)."""
    src_ck = latest_checkpoint(src_run)
    dst_ck = latest_checkpoint(dst_run) if os.path.isdir(dst_run) else None
    if dst_ck is not None and ckpt_step(dst_ck) >= ckpt_step(src_ck):
        log(f"[kaggle] local checkpoint {dst_ck} is already as new as {src_ck}")
        return dst_ck
    os.makedirs(os.path.join(dst_run, "ckpt"), exist_ok=True)
    name = os.path.basename(src_ck)
    shutil.copytree(src_ck, os.path.join(dst_run, "ckpt", name), dirs_exist_ok=True)
    with open(os.path.join(dst_run, "ckpt", "latest"), "w") as f:
        f.write(name)
    for fn in ("metrics.jsonl", "config.yaml", "arch.json"):
        if os.path.exists(os.path.join(src_run, fn)):
            shutil.copy2(os.path.join(src_run, fn), os.path.join(dst_run, fn))
    log(f"[kaggle] restored {src_ck} (step {ckpt_step(src_ck)})")
    return os.path.join(dst_run, "ckpt", name)


def restore_data(input_root, rel_data, dst, log=print):
    if os.path.exists(os.path.join(dst, "manifest.json")):
        return True
    for man in glob.glob(os.path.join(input_root, "**", rel_data, "manifest.json"), recursive=True):
        shutil.copytree(os.path.dirname(man), dst, dirs_exist_ok=True)
        log(f"[kaggle] restored data from {os.path.dirname(man)}")
        return True
    return False


def minutes_left(session_start, budget_hours, margin_min):
    return budget_hours * 60 - (time.time() - session_start) / 60 - margin_min


def run(cmd, log=print):
    log("[kaggle] $ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specialist", required=True, choices=["tricode", "trimath", "trireason"])
    p.add_argument("--preset", default="partial", choices=["partial", "quick"])
    p.add_argument("--config", default=None, help="override the config path (default: the preset's kaggle_t4 file)")
    p.add_argument("--input-root", default="/kaggle/input")
    p.add_argument("--session-start", type=float, default=float(os.environ.get("TRIFORGE_SESSION_START", time.time())),
                   help="epoch seconds when the Kaggle session started (the notebook's first cell sets it)")
    p.add_argument("--budget-hours", type=float, default=11.5, help="usable part of Kaggle's 12-hour limit")
    p.add_argument("--save-margin-min", type=float, default=20,
                   help="time kept free for the final evaluation, the checkpoint and Kaggle's output upload")
    p.add_argument("--export-when-done", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="extra dotted overrides passed to train.py")
    args = p.parse_args()

    cfg_path = args.config or f"configs/{args.specialist}/bitnet2b_{args.preset}_kaggle_t4.yaml"
    cfg = load_config(os.path.join(ROOT, cfg_path), args.set)
    rel_out, rel_data = cfg["run"]["output_dir"], cfg["data"]["dir"]
    out_dir, data_dir = resolve_path(rel_out), resolve_path(rel_data)

    src_run, step = find_previous_run(args.input_root, rel_out)
    if src_run:
        restore_run(src_run, out_dir)
    else:
        print(f"[kaggle] no previous checkpoint of {rel_out} under {args.input_root}: starting fresh")

    if not restore_data(args.input_root, rel_data, data_dir):
        rc = run([sys.executable, "scripts/prepare_data.py", "--config", cfg_path])
        if rc != 0:
            sys.exit(rc)

    ck = latest_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    done = ck is not None and ckpt_step(ck) >= cfg["optim"]["max_steps"]
    if not done:
        left = minutes_left(args.session_start, args.budget_hours, args.save_margin_min)
        cap = cfg["runtime"].get("max_train_minutes")
        cap = left if cap is None else min(cap, left)
        if cap < 5:
            print(f"[kaggle] only {left:.0f} min left in this session: not starting training")
            sys.exit(0)
        print(f"[kaggle] training for at most {cap:.0f} min")
        rc = run([sys.executable, "scripts/train.py", "--config", cfg_path, "--resume", "auto",
                  "--set", f"runtime.max_train_minutes={cap:.1f}", *args.set])
        if rc != 0:
            sys.exit(rc)
        ck = latest_checkpoint(out_dir)
        done = ckpt_step(ck) >= cfg["optim"]["max_steps"]

    print(f"[kaggle] checkpoint {ck}, step {ckpt_step(ck)} of {cfg['optim']['max_steps']}"
          + (" (schedule finished)" if done else ": run this notebook again with this output as input"))
    if done and args.export_when_done:
        exp = os.path.join("checkpoints", "exports", f"{args.specialist}-2b-{args.preset}-kaggle")
        sys.exit(run([sys.executable, "scripts/export.py", "--run", rel_out, "--out", exp]))


if __name__ == "__main__":
    main()

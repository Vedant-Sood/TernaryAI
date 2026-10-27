"""Run one session of a 2B TriForge fine-tune on a free cloud GPU, resuming where the last one stopped.

    # Kaggle: the previous session's output is added to the notebook as an input
    python scripts/cloud_session.py --specialist trimath --preset partial --input-root /kaggle/input
    # Colab: checkpoints live in Google Drive, restored at the start and synced during training
    # Modal: see modal/train_modal.py (checkpoints are written straight to a Modal Volume)
    python scripts/cloud_session.py --specialist trimath --preset partial \
        --input-root /content/drive/MyDrive/TriForge --sync-dir /content/drive/MyDrive/TriForge

Free notebooks are time-limited (Kaggle 12 h, Colab up to ~12 h and often less), so a long run is a
chain of sessions. Each session:

  1. restores the newest checkpoint (and the tokenized data) found under --input-root,
  2. prepares the data if no copy was found,
  3. trains with `--resume auto` and a wall-clock cap computed from the time left in the session,
     so the run evaluates and saves before the platform stops it,
  4. with --sync-dir, copies every new checkpoint there while training (Colab can disconnect
     without warning, and only the Drive copy survives),
  5. optionally exports the model once the schedule is finished.

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


def _copy_side_files(src_run, dst_run):
    for fn in ("metrics.jsonl", "config.yaml", "arch.json"):
        if os.path.exists(os.path.join(src_run, fn)):
            shutil.copy2(os.path.join(src_run, fn), os.path.join(dst_run, fn))


def _discard(path):
    """Delete a directory, truncating its files first: Google Drive keeps deleted files in the
    trash, where they still count against the 15 GB quota, but an emptied file costs nothing."""
    for d, _, files in os.walk(path):
        for fn in files:
            try:
                open(os.path.join(d, fn), "wb").close()
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def copy_run(src_run, dst_run, log=print, prune=False):
    """Copy the newest checkpoint of src_run (and its metrics) into dst_run unless dst_run already
    has one at least as new. The copy is atomic (temp dir + rename, `latest` written last). With
    prune, older checkpoints in dst_run are discarded. Returns the checkpoint now in dst_run."""
    src_ck = latest_checkpoint(src_run)
    if src_ck is None:
        return None
    dst_ck = latest_checkpoint(dst_run) if os.path.isdir(dst_run) else None
    name = os.path.basename(src_ck)
    ckdir = os.path.join(dst_run, "ckpt")
    if dst_ck is None or ckpt_step(dst_ck) < ckpt_step(src_ck):
        os.makedirs(ckdir, exist_ok=True)
        tmp = os.path.join(ckdir, f".tmp_{name}")
        _discard(tmp)
        try:
            shutil.copytree(src_ck, tmp)
        except (FileNotFoundError, shutil.Error):  # the trainer pruned it mid-copy; next one wins
            _discard(tmp)
            return dst_ck
        _discard(os.path.join(ckdir, name))
        os.replace(tmp, os.path.join(ckdir, name))
        with open(os.path.join(ckdir, "latest.tmp"), "w") as f:
            f.write(name)
        os.replace(os.path.join(ckdir, "latest.tmp"), os.path.join(ckdir, "latest"))
        dst_ck = os.path.join(ckdir, name)
        log(f"[session] copied {src_ck} (step {ckpt_step(src_ck)}) -> {dst_ck}")
    _copy_side_files(src_run, dst_run)
    if prune:
        keep = os.path.basename(dst_ck)
        for old in os.listdir(ckdir):
            if old.startswith(("step_", ".tmp_")) and old != keep:
                _discard(os.path.join(ckdir, old))
    return dst_ck


def restore_data(input_root, rel_data, dst, log=print):
    if os.path.exists(os.path.join(dst, "manifest.json")):
        return True
    for man in glob.glob(os.path.join(input_root, "**", rel_data, "manifest.json"), recursive=True):
        shutil.copytree(os.path.dirname(man), dst, dirs_exist_ok=True)
        log(f"[session] restored data from {os.path.dirname(man)}")
        return True
    return False


def minutes_left(session_start, budget_hours, margin_min):
    return budget_hours * 60 - (time.time() - session_start) / 60 - margin_min


def run(cmd, log=print):
    log("[session] $ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def train_with_sync(cmd, out_dir, sync_run, poll_s=60, log=print):
    """Run the trainer; while it runs, copy each new checkpoint to sync_run."""
    log("[session] $ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT)
    while proc.poll() is None:
        time.sleep(poll_s)
        if sync_run and os.path.isdir(out_dir):
            copy_run(out_dir, sync_run, log=log, prune=True)
    if sync_run:
        copy_run(out_dir, sync_run, log=log, prune=True)
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specialist", required=True, choices=["tricode", "trimath", "trireason"])
    p.add_argument("--preset", default="partial", choices=["partial", "quick"])
    p.add_argument("--profile", default="t4", choices=["t4", "l4"],
                   help="t4: 16 GB, fp16 (Kaggle, Colab); l4: 24 GB with bf16 (L4, A10G, e.g. Modal)")
    p.add_argument("--config", default=None, help="override the config path (default: bitnet2b_<preset>_<profile>.yaml)")
    p.add_argument("--input-root", default="/kaggle/input", help="where earlier sessions' outputs are")
    p.add_argument("--sync-dir", default=None,
                   help="mirror checkpoints, data and exports here during the session (e.g. a Google Drive folder)")
    p.add_argument("--session-start", type=float, default=float(os.environ.get("TRIFORGE_SESSION_START", time.time())),
                   help="epoch seconds when the notebook session started (the notebook's first cell sets it)")
    p.add_argument("--budget-hours", type=float, default=11.5, help="usable part of the platform's session limit")
    p.add_argument("--save-margin-min", type=float, default=20,
                   help="time kept free for the final evaluation, the checkpoint and the output upload")
    p.add_argument("--export-when-done", action="store_true")
    p.add_argument("--set", nargs="*", default=[], help="extra dotted overrides passed to train.py")
    args = p.parse_args()

    cfg_path = args.config or f"configs/{args.specialist}/bitnet2b_{args.preset}_{args.profile}.yaml"
    cfg = load_config(os.path.join(ROOT, cfg_path), args.set)
    rel_out, rel_data = cfg["run"]["output_dir"], cfg["data"]["dir"]
    out_dir, data_dir = resolve_path(rel_out), resolve_path(rel_data)
    sync_run = os.path.join(args.sync_dir, rel_out) if args.sync_dir else None

    src_run, step = find_previous_run(args.input_root, rel_out)
    if src_run:
        copy_run(src_run, out_dir)
    else:
        print(f"[session] no previous checkpoint of {rel_out} under {args.input_root}: starting fresh")

    if not restore_data(args.input_root, rel_data, data_dir):
        rc = run([sys.executable, "scripts/prepare_data.py", "--config", cfg_path])
        if rc != 0:
            sys.exit(rc)
    if args.sync_dir and not os.path.exists(os.path.join(args.sync_dir, rel_data, "manifest.json")):
        shutil.copytree(data_dir, os.path.join(args.sync_dir, rel_data), dirs_exist_ok=True)

    ck = latest_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    done = ck is not None and ckpt_step(ck) >= cfg["optim"]["max_steps"]
    if not done:
        left = minutes_left(args.session_start, args.budget_hours, args.save_margin_min)
        cap = cfg["runtime"].get("max_train_minutes")
        cap = left if cap is None else min(cap, left)
        if cap < 5:
            print(f"[session] only {left:.0f} min left in this session: not starting training")
            sys.exit(0)
        print(f"[session] training for at most {cap:.0f} min")
        rc = train_with_sync([sys.executable, "scripts/train.py", "--config", cfg_path, "--resume", "auto",
                              "--set", f"runtime.max_train_minutes={cap:.1f}", *args.set], out_dir, sync_run)
        if rc != 0:
            sys.exit(rc)
        ck = latest_checkpoint(out_dir)
        done = ckpt_step(ck) >= cfg["optim"]["max_steps"]

    print(f"[session] checkpoint {ck}, step {ckpt_step(ck)} of {cfg['optim']['max_steps']}"
          + (" (schedule finished)" if done else ": run the notebook again to continue"))
    if done and args.export_when_done:
        exp = os.path.join("checkpoints", "exports", f"{args.specialist}-2b-{args.preset}-{args.profile}")
        rc = run([sys.executable, "scripts/export.py", "--run", rel_out, "--out", exp])
        if rc == 0 and args.sync_dir:
            shutil.copytree(os.path.join(ROOT, exp), os.path.join(args.sync_dir, exp), dirs_exist_ok=True)
        sys.exit(rc)


if __name__ == "__main__":
    main()

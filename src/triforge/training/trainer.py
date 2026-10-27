"""Quantization-aware training loop (from scratch, fine-tuning, or distillation).

    python scripts/train.py --config configs/trimath/cpu_validation.yaml
    python scripts/train.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto

Features: bf16 autocast on CUDA (fp16 + loss scaling on GPUs without native bf16, e.g. the T4),
gradient accumulation, gradient checkpointing, sequence-length
schedule, cosine or WSD learning rate, progressive ternarization, optional distillation teacher,
atomic resumable checkpoints (model + optimizer + RNG + data sampler), JSONL metrics, and an
out-of-memory handler that writes a diagnostic report instead of dying silently.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import yaml

from ..datasets.loader import TokenDataset
from ..quantization.bitlinear import set_ternary_mix
from ..quantization.convert import flip_rate, snapshot_codes, ternary_report
from .build import build_model, load_teacher, save_arch, trainable_summary
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import get, resolve_path
from .optim import build_optimizer


def lr_at(step, ocfg):
    peak, warm, total = ocfg["lr"], ocfg.get("warmup_steps", 100), ocfg["max_steps"]
    min_lr = peak * ocfg.get("min_lr_frac", 0.1)
    if step < warm:
        return peak * (step + 1) / warm
    if ocfg.get("schedule", "cosine") == "wsd":  # warmup-stable-decay; decay over the last 20%
        decay_start = int(total * (1 - ocfg.get("decay_frac", 0.2)))
        if step < decay_start:
            return peak
        return min_lr + (peak - min_lr) * max(0.0, (total - step) / max(1, total - decay_start))
    progress = (step - warm) / max(1, total - warm)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def seq_len_at(step, rcfg):
    sched = rcfg.get("seq_len_schedule")
    if not sched:
        return rcfg["seq_len"]
    cur = sched[0][1]
    for s, L in sched:
        if step >= s:
            cur = L
    return cur


def peak_memory_gib(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 ** 3
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except ImportError:
        return float("nan")


def native_bf16(device) -> bool:
    """bf16 tensor cores need compute capability 8.0+ (Ampere); a T4 or P100 only emulates bf16."""
    return torch.cuda.get_device_capability(device)[0] >= 8


class Trainer:
    def __init__(self, cfg: dict, log=print):
        self.cfg, self.log = cfg, log
        run, rcfg = cfg["run"], cfg["runtime"]
        self.out = resolve_path(run["output_dir"])
        os.makedirs(self.out, exist_ok=True)
        dev = rcfg.get("device", "auto")
        self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available() else
                                   ("cpu" if dev == "auto" else dev))
        if rcfg.get("threads"):
            torch.set_num_threads(rcfg["threads"])
        torch.manual_seed(run.get("seed", 1337))
        prec = rcfg.get("precision", "auto")
        if self.device.type == "cuda" and prec == "auto":
            prec = "bf16" if native_bf16(self.device) else "fp16"
        self.amp_dtype = None
        if self.device.type == "cuda" and prec == "bf16":
            self.amp_dtype = torch.bfloat16
        elif prec == "fp16":
            self.amp_dtype = torch.float16
        self.model = build_model(cfg["model"], device=self.device)
        pdt = rcfg.get("param_dtype")
        if self.device.type == "cuda" and pdt == "bf16":
            self.model.to(torch.bfloat16)
        elif pdt == "fp32":
            # trainable weights in fp32 (frozen packed blocks and embeddings keep their dtype):
            # needed for fp16 loss scaling, and small QAT updates are not rounded away
            for p in self.model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        # fp16 has a narrow exponent range: scale the loss so small gradients do not underflow
        self.scaler = torch.amp.GradScaler(self.device.type) if self.amp_dtype == torch.float16 else None
        self.teacher = load_teacher(cfg["distill"], self.device) if get(cfg, "distill.enabled") else None
        if self.teacher is None and rcfg.get("loss_chunk") and hasattr(self.model, "loss_chunk"):
            self.model.loss_chunk = rcfg["loss_chunk"]
        data_dir = resolve_path(cfg["data"]["dir"])
        self.train_ds = TokenDataset(data_dir, "train", seed=run.get("seed", 1337))
        self.val_ds = TokenDataset(data_dir, "val", seed=0)
        self.opt = build_optimizer(self.model, cfg["optim"])
        self.step, self.tokens_seen = 0, 0
        self.metrics_path = os.path.join(self.out, "metrics.jsonl")

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        if self.amp_dtype is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _write(self, rec):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _loss(self, x, y, m):
        with self._autocast():
            logits, loss = self.model(x, y, loss_mask=m)
            if self.teacher is not None:
                from ..distillation import kd_loss
                with torch.no_grad():
                    t_logits, _ = self.teacher(x)
                d = self.cfg["distill"]
                loss, parts = kd_loss(logits, t_logits, y, m, alpha=d.get("alpha", 0.5),
                                      temperature=d.get("temperature", 2.0))
        return loss

    @torch.no_grad()
    def evaluate(self, iters=None):
        ecfg, rcfg = self.cfg.get("eval", {}), self.cfg["runtime"]
        iters = iters or ecfg.get("val_iters", 20)
        self.model.eval()
        g = torch.Generator().manual_seed(0)
        losses = []
        L = rcfg.get("eval_seq_len", rcfg["seq_len"])
        for _ in range(iters):
            x, y, m = self.val_ds.batch(rcfg.get("eval_batch_size", rcfg["micro_batch_size"]), L,
                                        self.device, generator=g)
            with self._autocast():
                _, loss = self.model(x, y, loss_mask=m)
            losses.append(loss.item())
        self.model.train()
        v = sum(losses) / len(losses)
        return {"val_loss": v, "val_ppl": math.exp(min(v, 50))}

    def save(self, extra=None):
        st = {"tokens_seen": self.tokens_seen, "data_state": self.train_ds.state().tolist(), **(extra or {})}
        if self.scaler is not None:
            st["grad_scaler"] = self.scaler.state_dict()
        p = save_checkpoint(self.out, self.step, self.model, self.opt, {"step": self.step}, st,
                            keep_last=self.cfg["runtime"].get("keep_last", 3))
        self.log(f"[ckpt] saved {p}")
        return p

    def resume(self, spec):
        path = latest_checkpoint(self.out) if spec == "auto" else spec
        if path is None:
            self.log("[resume] no checkpoint found, starting fresh")
            return
        st = load_checkpoint(path, self.model, self.opt, map_location=self.device)
        self.step, self.tokens_seen = st["step"], st.get("tokens_seen", 0)
        if "data_state" in st:
            self.train_ds.set_state(torch.tensor(st["data_state"], dtype=torch.uint8))
        if self.scaler is not None and "grad_scaler" in st:
            self.scaler.load_state_dict(st["grad_scaler"])
        self.log(f"[resume] from {path} at step {self.step}")

    # ------------------------------------------------------------------ main loop
    def fit(self):
        cfg, ocfg, rcfg, ecfg = self.cfg, self.cfg["optim"], self.cfg["runtime"], self.cfg.get("eval", {})
        with open(os.path.join(self.out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        save_arch(self.model, self.out)
        summary = {**trainable_summary(self.model), **ternary_report(self.model),
                   "device": str(self.device), "amp": str(self.amp_dtype)}
        self.log(f"[model] {json.dumps(summary)}")
        self._write({"event": "start", "step": self.step, **summary})
        accum = rcfg.get("grad_accum_steps", 1)
        mb = rcfg["micro_batch_size"]
        ptern = ocfg.get("progressive_ternarization_steps", 0)
        # Fraction of ternary codes that changed since the start of this run: shows how much QAT
        # actually moves a ternary model (a small LR or a LoRA delta may flip almost nothing).
        codes0 = snapshot_codes(self.model) if rcfg.get("track_flips", True) else None
        self.model.train()
        t_last, tok_last, t_start = time.time(), self.tokens_seen, time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        # Optional wall-clock cap (runtime.max_train_minutes): stop early and still evaluate and save.
        max_min = rcfg.get("max_train_minutes")
        stopped_on_time = False
        while self.step < ocfg["max_steps"]:
            if max_min is not None and time.time() - t_start >= 60 * max_min:
                stopped_on_time = True
                self.log(f"[train] time cap of {max_min} min reached at step {self.step}")
                break
            if self.step % ecfg.get("interval", 250) == 0:
                ev = self.evaluate()
                if codes0:
                    ev["flip_rate"] = flip_rate(codes0, self.model)
                self.log(f"[eval] step {self.step} val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f}")
                self._write({"event": "eval", "step": self.step, **ev})
            lr = lr_at(self.step, ocfg)
            for g in self.opt.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            if ptern:
                set_ternary_mix(self.model, min(1.0, self.step / ptern))
            L = seq_len_at(self.step, rcfg)
            self.opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(accum):
                x, y, m = self.train_ds.batch(mb, L, self.device)
                loss = self._loss(x, y, m) / accum
                (loss if self.scaler is None else self.scaler.scale(loss)).backward()
                loss_acc += loss.item()
                self.tokens_seen += x.numel()
            if self.scaler is not None:
                self.scaler.unscale_(self.opt)
            gn = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                ocfg.get("grad_clip", 1.0))
            if self.scaler is None:
                self.opt.step()
            else:  # skips the update when the scaled gradients overflowed, then adjusts the scale
                self.scaler.step(self.opt)
                self.scaler.update()
            self.step += 1
            if not math.isfinite(loss_acc):
                self.save({"reason": "non-finite loss"})
                raise FloatingPointError(f"loss became {loss_acc} at step {self.step}")
            if self.step % rcfg.get("log_interval", 20) == 0:
                now = time.time()
                rec = {"event": "train", "step": self.step, "loss": loss_acc, "lr": lr, "grad_norm": float(gn),
                       "seq_len": L, "tokens_per_s": (self.tokens_seen - tok_last) / (now - t_last),
                       "steps_per_s": rcfg.get("log_interval", 20) / (now - t_last),
                       "peak_mem_gib": peak_memory_gib(self.device), "elapsed_s": now - t_start}
                if self.scaler is not None:
                    rec["grad_scale"] = self.scaler.get_scale()
                self._write(rec)
                self.log(f"[train] step {self.step} loss {loss_acc:.4f} lr {lr:.2e} "
                         f"{rec['tokens_per_s']:.0f} tok/s mem {rec['peak_mem_gib']:.2f} GiB")
                t_last, tok_last = now, self.tokens_seen
            if self.step % rcfg.get("checkpoint_interval", 500) == 0:
                self.save()
        ev = self.evaluate()
        if codes0:
            ev["flip_rate"] = flip_rate(codes0, self.model)
            ev["zero_fraction"] = ternary_report(self.model)["zero_fraction_of_ternary"]
        self.log(f"[eval] final step {self.step} val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.2f}")
        self._write({"event": "eval", "step": self.step, "final": True, "stopped_on_time": stopped_on_time, **ev,
                     "train_time_s": time.time() - t_start, "peak_mem_gib": peak_memory_gib(self.device)})
        self.save({"final": True})
        return ev


def oom_report(trainer_or_cfg, err, out_dir):
    """Write diagnostics for a CUDA out-of-memory failure and print actionable advice."""
    rep = {"error": str(err).splitlines()[0]}
    if torch.cuda.is_available():
        rep["memory_summary"] = torch.cuda.memory_summary()
        rep["max_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        free, total = torch.cuda.mem_get_info()
        rep["free_gib"], rep["total_gib"] = free / 1024 ** 3, total / 1024 ** 3
    cfg = trainer_or_cfg.cfg if hasattr(trainer_or_cfg, "cfg") else trainer_or_cfg
    rep["runtime"] = cfg.get("runtime")
    rep["model"] = {k: v for k, v in cfg.get("model", {}).items() if k != "arch"}
    rep["advice"] = [
        "Lower runtime.micro_batch_size and raise runtime.grad_accum_steps (same effective batch).",
        "Enable model.gradient_checkpointing.",
        "Shorten runtime.seq_len or add a runtime.seq_len_schedule.",
        "Use optim.name: adamw8bit or adamw_cpu.",
        "For BitNet 2B: train fewer blocks (model.trainable_layers) or use model.strategy: lora.",
        "Run `python scripts/profile_hardware.py --config <cfg>` to see the estimated breakdown.",
        "The model size was NOT changed automatically.",
    ]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "oom_report.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\nCUDA out of memory. Diagnostics written to {path}\n  - " + "\n  - ".join(rep["advice"]),
          file=sys.stderr)
    return path

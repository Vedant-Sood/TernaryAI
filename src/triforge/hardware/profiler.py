"""Hardware profiler: detect the machine and pick a safe initial runtime configuration.

The profiler only chooses *runtime* knobs (micro-batch, sequence length, gradient
checkpointing, optimizer placement). It never shrinks the model or changes the architecture;
if even the smallest runtime setting does not fit, it raises `DoesNotFit` with a breakdown.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field

import torch

from .memory import GiB, estimate_training_memory


@dataclass
class HardwareInfo:
    os: str
    python: str
    torch: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None = None
    gpu_total_gib: float = 0.0
    gpu_free_gib: float = 0.0
    compute_capability: tuple | None = None
    bf16: bool = False
    cpu_count: int = 1
    ram_total_gib: float = 0.0
    ram_available_gib: float = 0.0
    disk_free_gib: float = 0.0
    wsl: bool = False
    notes: list = field(default_factory=list)


def detect(path=".") -> HardwareInfo:
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_t, ram_a = vm.total / GiB, vm.available / GiB
    except ImportError:
        ram_t = ram_a = 0.0
    rel = platform.release().lower()
    info = HardwareInfo(
        os=f"{platform.system()} {platform.release()}", python=platform.python_version(),
        torch=torch.__version__, cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
        cpu_count=os.cpu_count() or 1, ram_total_gib=ram_t, ram_available_gib=ram_a,
        disk_free_gib=shutil.disk_usage(path).free / GiB, wsl="microsoft" in rel or "wsl" in rel,
    )
    if info.cuda_available:
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.gpu_name, info.gpu_total_gib, info.gpu_free_gib = p.name, total / GiB, free / GiB
        info.compute_capability = (p.major, p.minor)
        info.bf16 = torch.cuda.is_bf16_supported()
        if info.gpu_total_gib < 9:
            info.notes.append("<=8 GB VRAM: train 2B models only with frozen packed blocks or ternary LoRA")
    else:
        info.notes.append("no CUDA device: CPU training is only practical for models below ~50M params")
    if platform.system() == "Windows" and "CONDA_PREFIX" in os.environ and \
            os.path.basename(os.environ["CONDA_PREFIX"]).lower() in ("anaconda3", "miniconda3", "base"):
        info.notes.append("running in conda base: pip torch + conda MKL can crash with OMP Error #15; "
                          "use a dedicated env (see docs/INSTALL.md)")
    return info


class DoesNotFit(RuntimeError):
    pass


def choose_runtime(model_cfg, info: HardwareInfo, strategy: dict, requested: dict) -> dict:
    """Pick micro-batch / sequence length that fit, honouring explicit user requests.

    requested keys: micro_batch_size, seq_len, gradient_checkpointing, optimizer (any may be "auto").
    """
    counts = model_cfg.param_counts()
    budget = (info.gpu_free_gib if info.cuda_available else info.ram_available_gib * 0.7) * 0.9 * GiB
    seqs = [requested["seq_len"]] if requested.get("seq_len", "auto") != "auto" else \
        [s for s in (4096, 2048, 1024, 512, 256) if s <= model_cfg.max_seq_len]
    mbs = [requested["micro_batch_size"]] if requested.get("micro_batch_size", "auto") != "auto" else \
        [16, 8, 4, 2, 1]
    gcs = [requested["gradient_checkpointing"]] if requested.get("gradient_checkpointing", "auto") != "auto" \
        else [False, True]
    opts = [requested["optimizer"]] if requested.get("optimizer", "auto") != "auto" else \
        ["adamw", "adamw8bit", "adamw_cpu"]
    tried = []
    for seq in seqs:
        for opt in opts:
            for gc in gcs:
                for mb in mbs:
                    est = estimate_training_memory(
                        counts, model_cfg.d_model, model_cfg.d_ff, model_cfg.n_layers, model_cfg.vocab_size,
                        mb, seq, optimizer=opt, grad_checkpointing=gc,
                        n_trainable_layers=strategy.get("n_trainable_layers"),
                        train_embeddings=strategy.get("train_embeddings", True),
                        lora_rank=strategy.get("lora_rank", 0),
                        param_bytes=2 if info.cuda_available else 4,
                        frozen_format=strategy.get("frozen_format", "packed"))
                    tried.append((seq, opt, gc, mb, est))
                    if est.total <= budget:
                        return {"seq_len": seq, "micro_batch_size": mb, "gradient_checkpointing": gc,
                                "optimizer": opt, "estimate_gib": est.as_gib(),
                                "budget_gib": round(budget / GiB, 2)}
    smallest = min(tried, key=lambda t: t[4].total)
    raise DoesNotFit(
        f"model {model_cfg.name} ({counts['total'] / 1e6:.0f}M params) does not fit in "
        f"{budget / GiB:.1f} GiB even at seq={smallest[0]}, micro_batch={smallest[3]}, "
        f"optimizer={smallest[1]}, checkpointing={smallest[2]}: needs {smallest[4].as_gib()}. "
        "The model was NOT shrunk. Options: freeze more blocks (strategy.n_trainable_layers), use "
        "ternary LoRA (strategy.lora_rank), CPU optimizer offload, or larger hardware.")

"""Optimizers. Weight decay is not applied to norms, embeddings, scales or LoRA B matrices."""

from __future__ import annotations

import torch


def param_groups(model, weight_decay, embed_lr_scale=1.0):
    decay, no_decay, embed = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in n or "lm_head" in n:
            embed.append(p)
        elif p.dim() >= 2 and "lora_B" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if embed:
        groups.append({"params": embed, "weight_decay": 0.0, "lr_scale": embed_lr_scale})
    return [g for g in groups if g["params"]]


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW with fp32 master weights and optimizer states in system RAM.

    GPU memory per trainable parameter drops to the bf16 weight + bf16 grad. Each step copies
    grads to the CPU, updates there, and copies weights back: slow, but it lets larger slices
    of a model train on an 8 GB card.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.inner_groups = []
        for g in self.param_groups:
            masters = [p.detach().float().cpu().pin_memory() if torch.cuda.is_available() else p.detach().float().cpu()
                       for p in g["params"]]
            for m in masters:
                m.requires_grad_(True)
            self.inner_groups.append(masters)
        self.inner = torch.optim.AdamW([{"params": m, "weight_decay": g["weight_decay"]}
                                        for m, g in zip(self.inner_groups, self.param_groups)],
                                       lr=lr, betas=betas, eps=eps)

    @torch.no_grad()
    def step(self, closure=None):
        for g, ig, masters in zip(self.param_groups, self.inner.param_groups, self.inner_groups):
            ig["lr"] = g["lr"]
            for p, m in zip(g["params"], masters):
                m.grad = None if p.grad is None else p.grad.detach().float().cpu()
        self.inner.step()
        for g, masters in zip(self.param_groups, self.inner_groups):
            for p, m in zip(g["params"], masters):
                p.copy_(m.to(p.device, non_blocking=True))

    def state_dict(self):
        return {"inner": self.inner.state_dict(), "masters": [[m.detach() for m in ms] for ms in self.inner_groups]}

    def load_state_dict(self, sd):
        self.inner.load_state_dict(sd["inner"])
        for ms, saved in zip(self.inner_groups, sd["masters"]):
            for m, s in zip(ms, saved):
                m.data.copy_(s)


def build_optimizer(model, ocfg):
    name = ocfg.get("name", "adamw")
    groups = param_groups(model, ocfg.get("weight_decay", 0.1), ocfg.get("embed_lr_scale", 1.0))
    betas = tuple(ocfg.get("betas", (0.9, 0.95)))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=ocfg["lr"], betas=betas, fused=torch.cuda.is_available() and
                                 all(p.is_cuda for g in groups for p in g["params"]))
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError("optim.name=adamw8bit needs `pip install bitsandbytes` (Linux/WSL2)") from e
        return bnb.optim.AdamW8bit(groups, lr=ocfg["lr"], betas=betas)
    if name == "adamw_cpu":
        return CPUOffloadAdamW(groups, lr=ocfg["lr"], betas=betas)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=ocfg["lr"], momentum=0.9)
    raise ValueError(f"unknown optimizer {name}")

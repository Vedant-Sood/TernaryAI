"""A small LLaMA-style decoder-only transformer.

The same architecture is used for both runs; only the linear layer changes:
  linear="ternary" -> every projection in attention and the MLP is a BitLinear
  linear="fp"      -> plain nn.Linear (full-precision baseline)

As in BitNet b1.58, the token embedding and the output head stay in full
precision; only the transformer blocks are ternary.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear, RMSNorm


@dataclass
class ModelConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    linear: str = "ternary"  # "ternary" or "fp"


def make_linear(cfg: ModelConfig, d_in: int, d_out: int) -> nn.Module:
    if cfg.linear == "ternary":
        return BitLinear(d_in, d_out, bias=False)
    if cfg.linear == "fp":
        return nn.Linear(d_in, d_out, bias=False)
    raise ValueError(f"unknown linear type: {cfg.linear}")


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = make_linear(cfg, cfg.n_embd, 3 * cfg.n_embd)
        self.proj = make_linear(cfg, cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    """SwiGLU feed-forward, hidden size ~ 8/3 * d (rounded to a multiple of 64)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(math.ceil(8 * cfg.n_embd / 3 / 64) * 64)
        self.gate = make_linear(cfg, cfg.n_embd, hidden)
        self.up = make_linear(cfg, cfg.n_embd, hidden)
        self.down = make_linear(cfg, hidden, cfg.n_embd)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        cos, sin = rope_cache(cfg.block_size, cfg.n_embd // cfg.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

    def param_breakdown(self):
        """Count parameters that are ternary vs full precision."""
        ternary = fp = 0
        for mod in self.modules():
            if isinstance(mod, BitLinear):
                ternary += mod.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        fp = total - ternary
        return {"total": total, "ternary": ternary, "full_precision": fp}

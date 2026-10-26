"""TriForgeLM: LLaMA-style decoder whose projections are ternary BitLinear layers.

One implementation covers both
  * the small from-scratch models (SwiGLU, BitLinear with internal RMSNorm), and
  * the BitNet b1.58 2B4T layout (GQA, squared-ReLU GLU, sub-layer norms, tied embeddings),
    so the 2B4T bf16 master weights can be loaded and quantization-aware fine-tuned here
    (see triforge.huggingface.bitnet_weights).

RoPE uses the rotate-half convention of Hugging Face LLaMA/BitNet checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..quantization.bitlinear import BitLinear, PackedTernaryLinear, RMSNorm
from .config import TriForgeConfig


def make_linear(cfg: TriForgeConfig, d_in: int, d_out: int) -> nn.Module:
    if cfg.linear == "fp":
        return nn.Linear(d_in, d_out, bias=False)
    return BitLinear(d_in, d_out, bias=False, input_norm=cfg.bitlinear_input_norm,
                     act_bits=cfg.act_bits, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha)


def rope_cache(seq_len, head_dim, theta, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos.to(x.dtype)) + (rotate_half(x) * sin.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, cfg: TriForgeConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        kv = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = make_linear(cfg, cfg.d_model, cfg.d_model)
        self.k_proj = make_linear(cfg, cfg.d_model, kv)
        self.v_proj = make_linear(cfg, cfg.d_model, kv)
        self.o_proj = make_linear(cfg, cfg.d_model, cfg.d_model)
        self.sub_norm = RMSNorm(cfg.d_model, cfg.norm_eps) if cfg.subln else None
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        past = 0
        if cache is not None:
            if cache.get("k") is not None:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        drop = self.dropout if self.training else 0.0
        if past == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, dropout_p=drop)
        else:
            # query i (absolute position past+i) may attend to keys 0..past+i
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        if self.sub_norm is not None:
            y = self.sub_norm(y)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TriForgeConfig):
        super().__init__()
        self.kind = cfg.mlp
        self.gate_proj = make_linear(cfg, cfg.d_model, cfg.d_ff)
        self.up_proj = make_linear(cfg, cfg.d_model, cfg.d_ff)
        self.down_proj = make_linear(cfg, cfg.d_ff, cfg.d_model)
        self.sub_norm = RMSNorm(cfg.d_ff, cfg.norm_eps) if cfg.subln else None

    def forward(self, x):
        g = self.gate_proj(x)
        g = F.silu(g) if self.kind == "swiglu" else F.relu(g).square()
        h = g * self.up_proj(x)
        if self.sub_norm is not None:
            h = self.sub_norm(h)
        return self.down_proj(h)


class Block(nn.Module):
    def __init__(self, cfg: TriForgeConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.drop(self.attn(self.input_norm(x), cos, sin, cache))
        return x + self.drop(self.mlp(self.post_attn_norm(x)))


class TriForgeLM(nn.Module):
    def __init__(self, cfg: TriForgeConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.loss_chunk = 0  # set >0 (e.g. 1024) to use chunked_lm_loss during training
        cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # scaled init for residual projections (GPT-2 style)
        for n, p in self.named_parameters():
            if n.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, caches=None, start_pos=0, loss_mask=None):
        T = idx.shape[1]
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence of {start_pos + T} exceeds max_seq_len={self.cfg.max_seq_len}")
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        x = self.embed(idx)
        for i, blk in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            if self.cfg.gradient_checkpointing and self.training and cache is None:
                x = checkpoint(blk, x, cos, sin, use_reentrant=False)
            else:
                x = blk(x, cos, sin, cache)
        h = self.final_norm(x)
        if targets is not None and self.loss_chunk and self.training:
            # Never materialise the full (B, T, vocab) fp32 logits: with a 128k vocabulary they
            # dominate training memory. Each chunk's logits are recomputed in backward.
            return None, chunked_lm_loss(self.lm_head, h, targets, loss_mask, self.loss_chunk)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = lm_loss(logits, targets, loss_mask)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None,
                 use_cache=True, stream=None):
        """Sample tokens. `stream(token_id)` is called for every new token if given."""
        self.eval()
        caches = [dict() for _ in self.blocks] if use_cache else None
        out = idx
        logits, _ = self(idx[:, -self.cfg.max_seq_len:], caches=caches)
        pos = min(idx.shape[1], self.cfg.max_seq_len)
        for _ in range(max_new_tokens):
            nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if stream is not None:
                stream(int(nxt[0]))
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if use_cache and pos < self.cfg.max_seq_len:
                logits, _ = self(nxt, caches=caches, start_pos=pos)
                pos += 1
            else:  # no cache, or window full: recompute on the last max_seq_len tokens
                caches = [dict() for _ in self.blocks] if use_cache else None
                window = out[:, -(self.cfg.max_seq_len - 1):] if use_cache else out[:, -self.cfg.max_seq_len:]
                logits, _ = self(window, caches=caches)
                pos = window.shape[1]
        return out

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def chunked_lm_loss(head, h, targets, loss_mask, chunk):
    B, T, D = h.shape
    hf, tf = h.reshape(-1, D), targets.reshape(-1)
    valid = (tf != -100).float()
    if loss_mask is not None:
        valid = valid * loss_mask.reshape(-1).float()

    def part(hc, tc, vc):
        lg = head(hc).float()
        return (F.cross_entropy(lg, tc.clamp(min=0), reduction="none") * vc).sum()

    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, hf.shape[0], chunk):
        total = total + checkpoint(part, hf[i:i + chunk], tf[i:i + chunk], valid[i:i + chunk], use_reentrant=False)
    return total / valid.sum().clamp(min=1.0)


def lm_loss(logits, targets, loss_mask=None):
    flat = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                           reduction="none", ignore_index=-100)
    if loss_mask is None:
        valid = (targets.reshape(-1) != -100).float()
    else:
        valid = loss_mask.reshape(-1).float() * (targets.reshape(-1) != -100).float()
    return (flat * valid).sum() / valid.sum().clamp(min=1.0)


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    if temperature is None or temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, -1).cumsum(-1)
        remove = probs - F.softmax(sorted_logits, -1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, -1), 1)


def ternary_linear_names(model: nn.Module):
    return [n for n, m in model.named_modules() if isinstance(m, (BitLinear, PackedTernaryLinear))]

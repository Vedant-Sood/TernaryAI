"""Architecture configuration for TriForge models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields


@dataclass
class TriForgeConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None        # None -> same as n_heads (no GQA)
    d_ff: int | None = None              # None -> 8/3 * d_model rounded up to a multiple of 64
    mlp: str = "swiglu"                  # "swiglu" | "relu2_glu" (BitNet 2B4T uses squared ReLU)
    subln: bool = False                  # extra RMSNorm before o_proj and down_proj (BitNet 2B4T)
    bitlinear_input_norm: bool = True    # RMSNorm inside every BitLinear (BitNet b1.58 paper recipe)
    linear: str = "ternary"              # "ternary" | "fp" (dense baseline with the same shapes)
    act_bits: int | None = 8
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    tie_embeddings: bool = False
    dropout: float = 0.0
    lora_rank: int = 0                   # ternary-preserving LoRA (see BitLinear)
    lora_alpha: float | None = None
    gradient_checkpointing: bool = False
    name: str = "triforge"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_ff is None:
            self.d_ff = int(math.ceil(8 * self.d_model / 3 / 64) * 64)
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError("d_model must divide by n_heads and n_heads by n_kv_heads")
        if self.mlp not in ("swiglu", "relu2_glu"):
            raise ValueError(f"unknown mlp {self.mlp}")
        if self.linear not in ("ternary", "fp"):
            raise ValueError(f"unknown linear {self.linear}")

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown architecture keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # Analytic parameter count, used by the memory estimator before anything is allocated.
    def param_counts(self) -> dict:
        d, f, L = self.d_model, self.d_ff, self.n_layers
        kv = self.n_kv_heads * self.head_dim
        attn = d * d + 2 * d * kv + d * d
        mlp = 3 * d * f
        linear_per_layer = attn + mlp
        norms_per_layer = 2 * d + (d + f if self.subln else 0)
        if self.bitlinear_input_norm and self.linear == "ternary":
            norms_per_layer += 3 * d + d + 2 * d + f  # q/k/v share input dim d; o: d; gate/up: d; down: f
        emb = self.vocab_size * d
        head = 0 if self.tie_embeddings else self.vocab_size * d
        linear = L * linear_per_layer
        other = L * norms_per_layer + emb + head + d
        return {
            "total": linear + other,
            "linear": linear,
            "ternary": linear if self.linear == "ternary" else 0,
            "embedding": emb,
            "head": head,
            "norms": L * norms_per_layer + d,
            "per_layer_linear": linear_per_layer,
        }

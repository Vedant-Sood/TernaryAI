"""Hugging Face Transformers models inside the TriForge trainer.

Used for Phase B (ternarizing *dense* pretrained models such as SmolLM2 or Qwen2.5) and as
dense teachers for distillation. HFAdapter gives an HF CausalLM the same call signature as
TriForgeLM: model(idx, targets, loss_mask=...) -> (logits, loss).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..architectures.transformer import lm_loss
from ..quantization.convert import ternarize_linears


@dataclass
class AdapterConfig:
    name: str
    vocab_size: int
    max_seq_len: int
    tie_embeddings: bool = False


class HFAdapter(nn.Module):
    def __init__(self, hf_model, name="hf"):
        super().__init__()
        self.hf = hf_model
        c = hf_model.config
        self.cfg = AdapterConfig(name=name, vocab_size=c.vocab_size,
                                 max_seq_len=getattr(c, "max_position_embeddings", 2048),
                                 tie_embeddings=getattr(c, "tie_word_embeddings", False))

    def forward(self, idx, targets=None, loss_mask=None, **_):
        logits = self.hf(input_ids=idx, use_cache=False).logits
        loss = lm_loss(logits, targets, loss_mask) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None, **_):
        do_sample = bool(temperature and temperature > 0)
        return self.hf.generate(idx, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                temperature=temperature if do_sample else None, top_k=top_k, top_p=top_p,
                                eos_token_id=eos_id, pad_token_id=eos_id)

    def gradient_checkpointing_enable(self):
        self.hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def load_dense(model_id: str, dtype=torch.bfloat16, revision=None, device="cpu"):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, revision=revision)
    return HFAdapter(m.to(device), name=model_id)


def ternarize_hf(adapter: HFAdapter, act_bits=8, lora_rank=0, skip=("*lm_head*", "*embed_tokens*")):
    """Post-training ternarization of every nn.Linear in the decoder (embeddings/head stay fp)."""
    names = ternarize_linears(adapter.hf, skip=skip, act_bits=act_bits, lora_rank=lora_rank)
    if lora_rank:
        for n, p in adapter.named_parameters():
            p.requires_grad_("lora_" in n)
    return names

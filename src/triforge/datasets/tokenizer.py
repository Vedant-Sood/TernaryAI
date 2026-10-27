"""Tokenizers and prompt templates.

Two kinds of tokenizer are supported behind one interface:
  * "bpe": a byte-level BPE trained locally with `tokenizers` (small from-scratch models; works offline)
  * "hf":  any Hugging Face tokenizer (e.g. the LLaMA-3 tokenizer of BitNet b1.58 2B4T)

Templates decide how an instruction example becomes tokens and which tokens are trained on
(loss mask = 1 only on the response, for supervised fine-tuning).
"""

from __future__ import annotations

import os

SPECIALS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]


class Tokenizer:
    def __init__(self, backend, kind: str, name: str):
        self.backend, self.kind, self.name = backend, kind, name
        if kind == "bpe":
            self.bos_id = backend.token_to_id("<|bos|>")
            self.eos_id = backend.token_to_id("<|eos|>")
            self.pad_id = backend.token_to_id("<|pad|>")
            self.vocab_size = backend.get_vocab_size()
        else:
            self.bos_id = backend.bos_token_id
            self.eos_id = backend.eos_token_id
            self.pad_id = backend.pad_token_id if backend.pad_token_id is not None else backend.eos_token_id
            self.vocab_size = len(backend)

    def encode(self, text: str) -> list[int]:
        if self.kind == "bpe":
            return self.backend.encode(text).ids
        return self.backend.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        if self.kind == "bpe":
            return self.backend.decode(list(ids), skip_special_tokens=True)
        return self.backend.decode(list(ids), skip_special_tokens=True)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        if self.kind == "bpe":
            self.backend.save(os.path.join(directory, "tokenizer.json"))
        else:
            self.backend.save_pretrained(directory)


def train_bpe(texts, vocab_size: int, path: str) -> Tokenizer:
    from tokenizers import Tokenizer as HFTok
    from tokenizers import decoders, models, pre_tokenizers, trainers
    tok = HFTok(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok.save(path)
    return Tokenizer(tok, "bpe", os.path.basename(path))


def load_tokenizer(spec: dict) -> Tokenizer:
    from ..training.config import resolve_path
    if spec["type"] == "bpe":
        from tokenizers import Tokenizer as HFTok
        p = resolve_path(spec["path"])
        if os.path.isdir(p):
            p = os.path.join(p, "tokenizer.json")
        return Tokenizer(HFTok.from_file(p), "bpe", os.path.basename(p))
    from transformers import AutoTokenizer
    return Tokenizer(AutoTokenizer.from_pretrained(spec["name"], revision=spec.get("revision")), "hf", spec["name"])


# --------------------------------------------------------------------------- templates

def format_example(tok: Tokenizer, ex: dict, template: str = "triforge", system: str | None = None):
    """Return (ids, loss_mask) for one example. Plain-text examples train on every token."""
    if "text" in ex:
        ids = tok.encode(ex["text"]) + [tok.eos_id]
        return ids, [1] * len(ids)
    if template == "triforge":
        prefix = f"<|user|>\n{ex['prompt']}\n<|assistant|>\n"
        p_ids = tok.encode(prefix)
        r_ids = tok.encode(ex["response"]) + [tok.eos_id]
    elif template == "bitnet":
        # BitNet b1.58 2B4T chat format (from its tokenizer_config chat_template)
        sys = f"System: {system}<|eot_id|>" if system else ""
        prefix = f"<|begin_of_text|>{sys}User: {ex['prompt']}<|eot_id|>Assistant: "
        p_ids = tok.encode(prefix)
        r_ids = tok.encode(ex["response"] + "<|eot_id|>")
    else:
        raise ValueError(f"unknown template {template}")
    return p_ids + r_ids, [0] * len(p_ids) + [1] * len(r_ids)


def prompt_ids(tok: Tokenizer, prompt: str, template: str = "triforge", system: str | None = None):
    if template == "triforge":
        return tok.encode(f"<|user|>\n{prompt}\n<|assistant|>\n")
    sys = f"System: {system}<|eot_id|>" if system else ""
    return tok.encode(f"<|begin_of_text|>{sys}User: {prompt}<|eot_id|>Assistant: ")

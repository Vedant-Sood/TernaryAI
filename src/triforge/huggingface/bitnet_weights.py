"""Load / save BitNet b1.58 2B4T weights in the TriForge layout.

Source checkpoint: microsoft/bitnet-b1.58-2B-4T-bf16 (MIT), the bf16 *master* weights that the
model card designates for training. (The non-bf16 repo holds packed inference weights.)
The model is natively ternary: trained from scratch with absmean ternary weights and 8-bit
activations, so quantization-aware fine-tuning keeps it in the regime it was trained in.

Loading streams tensors one at a time with safetensors.safe_open. Blocks that will stay
frozen are converted to packed 2-bit codes immediately, so the full bf16 model never has to
sit in GPU memory.
"""

from __future__ import annotations

import json
import os
import re

import torch

from ..architectures import TriForgeConfig, TriForgeLM
from ..quantization.bitlinear import BitLinear, PackedTernaryLinear, freeze_to_packed

HF_TO_TF = [
    (r"^model\.embed_tokens\.weight$", "embed.weight"),
    (r"^model\.norm\.weight$", "final_norm.weight"),
    (r"^lm_head\.weight$", "lm_head.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"blocks.\1.input_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blocks.\1.post_attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$", r"blocks.\1.attn.\2_proj.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.attn_sub_norm\.weight$", r"blocks.\1.attn.sub_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", r"blocks.\1.mlp.\2_proj.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.ffn_sub_norm\.weight$", r"blocks.\1.mlp.sub_norm.weight"),
]


def hf_to_triforge_name(name: str) -> str | None:
    for pat, rep in HF_TO_TF:
        if re.match(pat, name):
            return re.sub(pat, rep, name)
    return None


def triforge_to_hf_name(name: str) -> str | None:
    inverse = [
        (r"^embed\.weight$", "model.embed_tokens.weight"),
        (r"^final_norm\.weight$", "model.norm.weight"),
        (r"^lm_head\.weight$", "lm_head.weight"),
        (r"^blocks\.(\d+)\.input_norm\.weight$", r"model.layers.\1.input_layernorm.weight"),
        (r"^blocks\.(\d+)\.post_attn_norm\.weight$", r"model.layers.\1.post_attention_layernorm.weight"),
        (r"^blocks\.(\d+)\.attn\.(q|k|v|o)_proj\.weight$", r"model.layers.\1.self_attn.\2_proj.weight"),
        (r"^blocks\.(\d+)\.attn\.sub_norm\.weight$", r"model.layers.\1.self_attn.attn_sub_norm.weight"),
        (r"^blocks\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", r"model.layers.\1.mlp.\2_proj.weight"),
        (r"^blocks\.(\d+)\.mlp\.sub_norm\.weight$", r"model.layers.\1.mlp.ffn_sub_norm.weight"),
    ]
    for pat, rep in inverse:
        if re.match(pat, name):
            return re.sub(pat, rep, name)
    return None


def config_from_hf(hf_cfg: dict, max_seq_len: int | None = None, **overrides) -> TriForgeConfig:
    if hf_cfg.get("hidden_act", "relu2") != "relu2":
        raise ValueError("expected a BitNet config with relu2 activation")
    rope_theta = hf_cfg.get("rope_theta") or (hf_cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0)
    cfg = dict(
        vocab_size=hf_cfg["vocab_size"], max_seq_len=max_seq_len or hf_cfg["max_position_embeddings"],
        d_model=hf_cfg["hidden_size"], n_layers=hf_cfg["num_hidden_layers"], n_heads=hf_cfg["num_attention_heads"],
        n_kv_heads=hf_cfg["num_key_value_heads"], d_ff=hf_cfg["intermediate_size"], mlp="relu2_glu", subln=True,
        bitlinear_input_norm=False, linear="ternary", act_bits=8, rope_theta=rope_theta,
        norm_eps=hf_cfg["rms_norm_eps"], tie_embeddings=hf_cfg.get("tie_word_embeddings", True),
        name="bitnet-b1.58-2B-4T")
    cfg.update(overrides)
    return TriForgeConfig(**cfg)


def resolve_local(model_id_or_path: str, revision: str | None = None) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id_or_path, revision=revision,
                             allow_patterns=["*.json", "*.safetensors", "LICENSE", "*.md"])


def load_bitnet(model_id_or_path: str, *, trainable_layers=None, lora_rank: int = 0, max_seq_len=None,
                dtype=torch.bfloat16, device="cpu", revision=None, gradient_checkpointing=True,
                train_embeddings=False, cache_frozen_codes=True) -> TriForgeLM:
    """Build a TriForgeLM from BitNet bf16 master weights.

    trainable_layers: None = all layers trainable; a list of block indices = only these keep
    latent bf16 weights, all other blocks are frozen as packed ternary codes.
    """
    from safetensors import safe_open
    path = resolve_local(model_id_or_path, revision)
    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = config_from_hf(hf_cfg, max_seq_len=max_seq_len, lora_rank=lora_rank,
                         gradient_checkpointing=gradient_checkpointing)
    with torch.device("meta"):
        model = TriForgeLM(cfg)
    model = model.to_empty(device="cpu").to(dtype)
    if cfg.tie_embeddings:  # to_empty() allocates shared parameters separately; re-tie them
        model.lm_head.weight = model.embed.weight
    # rebuild non-persistent RoPE buffers that were created on meta
    from ..architectures.transformer import rope_cache
    cos, sin = rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
    model.rope_cos, model.rope_sin = cos, sin
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.ternary_mix = torch.tensor(1.0)
            if m.lora_rank:
                torch.nn.init.kaiming_uniform_(m.lora_A, a=5 ** 0.5)
                torch.nn.init.zeros_(m.lora_B)
    params = dict(model.named_parameters())
    loaded = set()
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    for fn in files:
        with safe_open(os.path.join(path, fn), framework="pt") as st:
            for k in st.keys():
                tk = hf_to_triforge_name(k)
                if tk is None or tk not in params:
                    continue
                with torch.no_grad():
                    params[tk].copy_(st.get_tensor(k).to(dtype))
                loaded.add(tk)
    expected = {n for n in params if "lora_" not in n}
    if cfg.tie_embeddings:
        expected.discard("lm_head.weight")
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"BitNet checkpoint is missing {len(missing)} tensors, e.g. {sorted(missing)[:3]}")

    if trainable_layers is not None and not lora_rank:
        keep = set(trainable_layers)
        for i, blk in enumerate(model.blocks):
            if i not in keep:
                freeze_to_packed(blk)
                for m in blk.modules():
                    if isinstance(m, PackedTernaryLinear):
                        m.cache_codes = cache_frozen_codes  # False: unpack 2-bit codes every forward
                for p in blk.parameters():
                    p.requires_grad_(False)
    if not train_embeddings:
        model.embed.weight.requires_grad_(False)
    if lora_rank:
        for n, p in model.named_parameters():
            p.requires_grad_("lora_" in n)
    return model.to(device)


def latent_equivalent(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """A latent weight that absmean-ternarizes back to exactly (codes, scale).

    With f = fraction of non-zero codes, W = codes * scale / f has mean|W| = scale and
    W / scale = codes / f, which rounds and clips back to codes for any f in (0, 1].
    """
    f = (codes != 0).float().mean().clamp(min=1e-6)
    return codes.float() * (scale.float() / f)


def save_hf_bf16(model: TriForgeLM, out_dir: str, hf_config_src: str):
    """Write the latent weights back in the HF BitNet layout (bf16, `quantization_mode: online`)."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    from ..quantization.bitlinear import PackedTernaryLinear
    tensors = {}
    for n, t in model.state_dict().items():
        hn = triforge_to_hf_name(n)
        if hn is None or (hn == "lm_head.weight" and model.cfg.tie_embeddings):
            continue
        tensors[hn] = t.detach().to(torch.bfloat16).contiguous().cpu()
    for n, m in model.named_modules():
        if isinstance(m, PackedTernaryLinear):
            tensors[triforge_to_hf_name(n + ".weight")] = latent_equivalent(m.codes(), m.scale).to(torch.bfloat16)
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(hf_config_src, "config.json")) as f:
        hf_cfg = json.load(f)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_cfg, f, indent=2)

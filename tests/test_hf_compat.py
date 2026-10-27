"""Hugging Face compatibility: TriForge's BitNet layout must reproduce transformers' BitNet."""

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import BitNetConfig, BitNetForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from triforge.huggingface.bitnet_weights import load_bitnet, save_hf_bf16  # noqa: E402
from triforge.huggingface.hf_models import HFAdapter, ternarize_hf  # noqa: E402
from triforge.quantization import BitLinear, ternary_report  # noqa: E402

TINY = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64, rope_theta=500000.0, rms_norm_eps=1e-5,
            hidden_act="relu2", tie_word_embeddings=True, bos_token_id=1, eos_token_id=2)


def _hf_bitnet(tmp_path, online_ternary: bool):
    torch.manual_seed(0)
    cfg = BitNetConfig(**TINY)
    hf = BitNetForCausalLM(cfg).float().eval()
    hf.save_pretrained(tmp_path, safe_serialization=True)
    with open(tmp_path / "config.json") as f:
        c = json.load(f)
    if online_ternary:
        c["quantization_config"] = {"quant_method": "bitnet", "linear_class": "autobitlinear",
                                    "quantization_mode": "online"}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(c, f)
    return hf


def test_bitnet_ternary_forward_matches_transformers(tmp_path):
    _hf_bitnet(tmp_path, online_ternary=True)
    ref = BitNetForCausalLM.from_pretrained(tmp_path, torch_dtype=torch.float32).eval()
    assert any(type(m).__name__ == "AutoBitLinear" for m in ref.modules())
    ours = load_bitnet(str(tmp_path), dtype=torch.float32, gradient_checkpointing=False).eval()
    x = torch.randint(0, 128, (2, 12))
    with torch.no_grad():
        a = ref(x).logits
        b, _ = ours(x)
    assert torch.allclose(a, b, atol=2e-4), (a - b).abs().max()


def test_bitnet_partial_freeze_and_hf_roundtrip(tmp_path):
    _hf_bitnet(tmp_path / "src", online_ternary=True)
    full = load_bitnet(str(tmp_path / "src"), dtype=torch.float32, gradient_checkpointing=False).eval()
    part = load_bitnet(str(tmp_path / "src"), trainable_layers=[1], dtype=torch.float32,
                       gradient_checkpointing=False).eval()
    assert not any(isinstance(m, BitLinear) for m in part.blocks[0].modules())
    assert any(isinstance(m, BitLinear) for m in part.blocks[1].modules())
    assert not any(p.requires_grad for p in part.blocks[0].parameters())
    x = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        assert torch.allclose(full(x)[0], part(x)[0], atol=1e-4)
    save_hf_bf16(part, str(tmp_path / "out"), str(tmp_path / "src"))
    back = BitNetForCausalLM.from_pretrained(tmp_path / "out", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        diff = (back(x).logits - full(x)[0]).abs().max()
    assert diff < 0.1  # bf16 storage of latents; ternary codes themselves are preserved exactly


def test_ternarize_dense_llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    ad = HFAdapter(LlamaForCausalLM(cfg).float())
    x = torch.randint(0, 100, (2, 16))
    dense_logits, _ = ad(x)
    names = ternarize_hf(ad)
    assert len(names) == 2 * 7 and not any("lm_head" in n for n in names)
    rep = ternary_report(ad)
    assert rep["ternary_params"] == 2 * (4 * 64 * 64 + 3 * 64 * 128)
    logits, loss = ad(x, x)
    assert logits.shape == dense_logits.shape and torch.isfinite(loss)
    loss.backward()
    assert ad.hf.model.layers[0].self_attn.q_proj.weight.grad is not None

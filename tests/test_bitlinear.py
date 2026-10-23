import torch

from compare import pack_2bit
from ternary import GPT, BitLinear, ModelConfig, activation_quant, weight_quant


def test_weight_quant_is_ternary():
    w = torch.randn(64, 32)
    wq = weight_quant(w)
    scale = w.abs().mean()
    levels = torch.unique((wq / scale).round())
    assert set(levels.tolist()) <= {-1.0, 0.0, 1.0}


def test_activation_quant_has_at_most_256_levels_per_token():
    x = torch.randn(4, 100)
    xq = activation_quant(x)
    for row, qrow in zip(x, xq):
        scale = 127.0 / row.abs().max()
        ints = (qrow * scale).round()
        assert ints.min() >= -128 and ints.max() <= 127
        assert torch.allclose(ints / scale, qrow, atol=1e-5)


def test_ste_passes_gradients_to_latent_weights():
    layer = BitLinear(16, 8)
    layer(torch.randn(2, 16)).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_ternary_export_matches_forward():
    layer = BitLinear(16, 8)
    q, s = layer.ternary_weight()
    assert torch.allclose(q.float() * s, weight_quant(layer.weight), atol=1e-6)


def test_pack_2bit_size():
    q = torch.randint(-1, 2, (1000,), dtype=torch.int8)
    assert len(pack_2bit(q)) == 250


def test_models_forward():
    for linear in ("ternary", "fp"):
        cfg = ModelConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, linear=linear)
        model = GPT(cfg)
        x = torch.randint(0, 50, (2, 16))
        logits, loss = model(x, x)
        assert logits.shape == (2, 16, 50) and torch.isfinite(loss)

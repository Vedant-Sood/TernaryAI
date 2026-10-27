import pytest
import torch

from triforge.architectures import TriForgeConfig, TriForgeLM
from triforge.quantization import BitLinear, ternary_report


def tiny(**kw):
    base = dict(vocab_size=97, max_seq_len=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=96)
    base.update(kw)
    return TriForgeConfig(**base)


@pytest.mark.parametrize("kw", [{}, {"mlp": "relu2_glu", "subln": True, "bitlinear_input_norm": False,
                                     "tie_embeddings": True}, {"linear": "fp"}])
def test_forward_backward_shapes_and_grads(kw):
    torch.manual_seed(0)
    m = TriForgeLM(tiny(**kw))
    x = torch.randint(0, 97, (3, 20))
    logits, loss = m(x, x)
    assert logits.shape == (3, 20, 97) and torch.isfinite(loss)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_param_count_formula_matches_module():
    for kw in [{}, {"subln": True, "bitlinear_input_norm": False, "tie_embeddings": True}, {"linear": "fp"}]:
        cfg = tiny(**kw)
        assert cfg.param_counts()["total"] == TriForgeLM(cfg).num_params(), kw


def test_ternary_fraction_reported():
    m = TriForgeLM(tiny())
    rep = ternary_report(m)
    n_lin = sum(mod.weight.numel() for mod in m.modules() if isinstance(mod, BitLinear))
    assert rep["ternary_params"] == n_lin and 0 < rep["ternary_fraction"] < 1
    assert 0.0 < rep["zero_fraction_of_ternary"] < 1.0


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(1)
    m = TriForgeLM(tiny(mlp="relu2_glu", subln=True, bitlinear_input_norm=False)).eval()
    x = torch.randint(0, 97, (1, 10))
    a = m.generate(x, 12, temperature=0.0, use_cache=True)
    b = m.generate(x, 12, temperature=0.0, use_cache=False)
    assert torch.equal(a, b)


def test_cache_prefill_with_existing_cache():
    torch.manual_seed(2)
    m = TriForgeLM(tiny()).eval()
    x = torch.randint(0, 97, (1, 16))
    full, _ = m(x)
    caches = [dict() for _ in m.blocks]
    m(x[:, :9], caches=caches)
    part, _ = m(x[:, 9:], caches=caches, start_pos=9)
    assert torch.allclose(full[:, 9:], part, atol=1e-4)


def test_gradient_checkpointing_equivalent():
    torch.manual_seed(3)
    a = TriForgeLM(tiny())
    b = TriForgeLM(tiny(gradient_checkpointing=True))
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, 97, (2, 16))
    la = a(x, x)[1]
    lb = b(x, x)[1]
    la.backward()
    lb.backward()
    assert torch.allclose(la, lb, atol=1e-6)
    ga = a.blocks[0].attn.q_proj.weight.grad
    gb = b.blocks[0].attn.q_proj.weight.grad
    assert torch.allclose(ga, gb, atol=1e-6)


def test_chunked_loss_equals_full_loss():
    torch.manual_seed(4)
    m = TriForgeLM(tiny())
    x = torch.randint(0, 97, (2, 24))
    mask = (torch.rand(2, 24) > 0.3).float()
    full = m(x, x, loss_mask=mask)[1]
    m.loss_chunk = 7
    m.train()
    logits, chunked = m(x, x, loss_mask=mask)
    assert logits is None and torch.allclose(full, chunked, atol=1e-5)


def test_sequence_too_long_raises():
    m = TriForgeLM(tiny())
    with pytest.raises(ValueError):
        m(torch.zeros(1, 65, dtype=torch.long))


def test_overfit_tiny_batch_converges():
    """Gradient flow sanity: a ternary model can memorise one batch."""
    torch.manual_seed(5)
    m = TriForgeLM(tiny())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 97, (4, 32))
    first = None
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3 * first

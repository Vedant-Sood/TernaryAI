import os

import pytest
import torch
import torch.nn.functional as F

from triforge.huggingface.bitnet_weights import latent_equivalent
from triforge.quantization import (BitLinear, PackedTernaryLinear, activation_quant, freeze_to_packed, pack,
                                   packed_nbytes, set_ternary_mix, ternarize, unpack, weight_quant)
from triforge.quantization import kernels


@pytest.mark.parametrize("shape", [(64, 32), (7, 13), (1, 1), (256, 1024)])
def test_ternarize_values_and_scale(shape):
    w = torch.randn(*shape) * 0.05
    t, gamma = ternarize(w)
    assert t.dtype == torch.int8
    assert set(torch.unique(t).tolist()) <= {-1, 0, 1}
    assert torch.isclose(gamma, w.abs().mean().clamp(min=1e-5))
    # forward weights are exactly gamma * T
    assert torch.allclose(weight_quant(w), t.float() * gamma, atol=1e-7)


def test_weight_quant_matches_bitnet_reference():
    w = torch.randn(48, 40)
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    ref = (w * scale).round().clamp(-1, 1) / scale  # transformers.integrations.bitnet.WeightQuant
    assert torch.allclose(weight_quant(w), ref, atol=1e-6)


def test_activation_quant_int8_levels():
    x = torch.randn(5, 64) * 3
    xq = activation_quant(x)
    s = 127.0 / x.abs().amax(-1, keepdim=True)
    ints = xq * s
    assert torch.allclose(ints, ints.round(), atol=1e-4)
    assert ints.min() >= -128 and ints.max() <= 127


def test_bitlinear_forward_weight_is_ternary_and_grads_flow():
    layer = BitLinear(32, 16, input_norm=True)
    w_fwd = layer.forward_weight().detach()
    t, g = layer.ternary_weight()
    assert torch.allclose(w_fwd, t.float() * g, atol=1e-7)
    x = torch.randn(4, 32, requires_grad=True)
    layer(x).pow(2).sum().backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0  # STE through activation quantization


@pytest.mark.parametrize("fmt", ["2bit", "base3"])
@pytest.mark.parametrize("n", [1, 4, 5, 999, 4096 + 3])
def test_pack_roundtrip(fmt, n):
    t = torch.randint(-1, 2, (n,), dtype=torch.int8)
    p = pack(t, fmt)
    assert p.dtype == torch.uint8 and p.numel() == packed_nbytes(n, fmt)
    assert torch.equal(unpack(p, (n,), fmt), t)


def test_pack_rejects_non_ternary():
    with pytest.raises(ValueError):
        pack(torch.tensor([0, 2], dtype=torch.int8))


@pytest.mark.parametrize("backend", ["reference", "int8", "addonly"])
def test_kernel_backends_match(backend):
    torch.manual_seed(0)
    t = torch.randint(-1, 2, (24, 40), dtype=torch.int8)
    gamma = torch.tensor(0.037)
    x = activation_quant(torch.randn(3, 5, 40))
    ref = F.linear(x, t.float() * gamma)
    out = kernels.ternary_linear(x, t, gamma, backend=backend)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not kernels.HAS_TRITON, reason="triton not installed")
def test_triton_kernel_in_interpreter():
    import subprocess
    import sys
    code = (
        "import torch, torch.nn.functional as F\n"
        "from triforge.quantization import kernels, pack\n"
        "t = torch.randint(-1, 2, (40, 72), dtype=torch.int8); g = torch.tensor(0.05)\n"
        "x = torch.randn(5, 72)\n"
        "y = kernels.ternary_linear(x, t, g, backend='triton', packed=pack(t))\n"
        "assert torch.allclose(y, F.linear(x, t.float() * g), atol=1e-4), (y - F.linear(x, t.float()*g)).abs().max()\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], env={**os.environ, "TRITON_INTERPRET": "1"},
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and ("interpret" in r.stderr.lower() and "not supported" in r.stderr.lower()):
        pytest.skip("triton interpreter unavailable")
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.parametrize("input_norm", [True, False])
def test_packed_layer_matches_bitlinear(input_norm):
    layer = BitLinear(40, 24, input_norm=input_norm)
    x = torch.randn(2, 7, 40)
    packed = PackedTernaryLinear.from_bitlinear(layer)
    assert torch.allclose(layer(x), packed(x), atol=1e-5)
    assert packed.packed.numel() == packed_nbytes(40 * 24)


def test_freeze_to_packed_saves_memory_and_keeps_input_grads():
    m = torch.nn.Sequential(BitLinear(64, 64), torch.nn.ReLU(), BitLinear(64, 8))
    x = torch.randn(3, 64)
    ref = m(x)
    saved = freeze_to_packed(m)
    assert saved > 0 and all(not isinstance(c, BitLinear) for c in m)
    x.requires_grad_(True)
    y = m(x)
    assert torch.allclose(y, ref, atol=1e-5)
    y.sum().backward()
    assert x.grad is not None


def test_lora_is_ternary_preserving_and_merges():
    layer = BitLinear(32, 16, lora_rank=4)
    assert not layer.weight.requires_grad and layer.lora_A.requires_grad
    with torch.no_grad():
        layer.lora_B.normal_(0, 0.05)
    x = torch.randn(3, 32)
    before = layer(x)
    w = layer.forward_weight().detach()
    t, g = layer.ternary_weight()
    assert torch.allclose(w, t.float() * g, atol=1e-7)  # forward weight is ternary even with adapters
    layer.merge_lora()
    assert layer.lora_rank == 0 and torch.allclose(layer(x), before, atol=1e-5)


def test_progressive_ternarization_mix():
    m = torch.nn.Sequential(BitLinear(16, 16))
    set_ternary_mix(m, 0.0)
    assert torch.allclose(m[0].forward_weight(), m[0].weight)
    set_ternary_mix(m, 1.0)
    assert torch.allclose(m[0].forward_weight(), weight_quant(m[0].weight))


def test_latent_equivalent_is_exact():
    for zero_frac in (0.1, 0.4, 0.7):
        t = torch.randint(-1, 2, (64, 64), dtype=torch.int8)
        t[torch.rand(64, 64) < zero_frac] = 0
        w = latent_equivalent(t, torch.tensor(0.02))
        t2, g2 = ternarize(w)
        assert torch.equal(t2, t) and torch.isclose(g2, torch.tensor(0.02))

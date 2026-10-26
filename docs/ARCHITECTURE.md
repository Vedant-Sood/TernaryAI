# TriForge architecture

## 1. What "ternary" means here

Every weight matrix inside the transformer blocks is used in the forward pass as

```
gamma = mean(|W|)                      one fp32 scale per matrix
T     = clip(round(W / gamma), -1, 1)  T ∈ {-1, 0, +1}
W_q   = gamma · T
```

This is BitNet b1.58's *absmean* quantizer ([Ma et al. 2024](https://arxiv.org/abs/2402.17764);
[2B4T report](https://arxiv.org/abs/2504.12285)). Activations entering each ternary layer are quantized
per token to int8 (*absmax*: `s = 127 / max|x|`). The combination is called **W1.58A8**.

`triforge.quantization.ternary` implements both quantizers. Tests check that:

* every forward weight is exactly `gamma · T` with `T ⊂ {-1,0,+1}` (`test_ternarize_values_and_scale`,
  `scripts/validate_ternary.py` for whole models),
* our quantizers are numerically identical to the ones in `transformers.integrations.bitnet`
  (`test_weight_quant_matches_bitnet_reference`), and
* a TriForge model loaded from BitNet weights reproduces `transformers`' own `BitNetForCausalLM` with
  online ternarization, to within 2e-4 on the logits (`test_bitnet_ternary_forward_matches_transformers`).

### Training: latent weights and the straight-through estimator

Rounding has zero gradient almost everywhere, so training keeps a higher-precision **latent** weight
`W` and uses the straight-through estimator (STE):

```
W_ste = W + stopgrad(Q(W) − W)    # forward value Q(W), gradient dL/dW = dL/dW_q
```

The same trick is used for activations. Many small updates accumulate in `W` until a code flips.
So the latent copy is **required during training**, and is why training memory is *not* small
(see [TRAINING.md](TRAINING.md#memory)). After training the latent copy is discarded.

### What stays in higher precision

| Component | Precision | Why |
| --- | --- | --- |
| Linear weights in attention and MLP (q, k, v, o, gate, up, down) | **ternary codes** + 1 fp32 scale per matrix | the ternary part: ~99% of block parameters |
| Latent weights during training | bf16 (GPU) or fp32 (CPU / offloaded optimizer) | needed by the STE; not exported |
| Token embedding and LM head (tied in 2B4T) | fp16 / bf16 | as in BitNet; ~14% of the 2B model's parameters |
| RMSNorm gains, sub-layer norms | fp16 / bf16 | tiny, numerically sensitive |
| Activations between layers, residual stream, attention scores, softmax, RoPE | bf16 / fp32 | only *inputs to ternary matmuls* are int8 |
| LoRA adapters (only with `strategy: lora`) | bf16, training only | merged into the latent weight and re-ternarized at export |

`ternary_report()` counts these exactly; e.g. for BitNet 2B4T: 2.413B parameters in total, 2.084B
(86.4%) ternary, 0.328B embedding, ~0.5M norm gains.

### Ternary is not the same as 4-bit / 8-bit quantization

| | Ternary (BitNet b1.58) | INT8 / INT4 PTQ (e.g. GPTQ, AWQ, bitsandbytes NF4) |
| --- | --- | --- |
| Values per weight | 3: {−γ, 0, +γ} | 256 or 16 levels |
| Scale granularity | one per matrix | per channel or per group of 32–128 |
| Arithmetic with codes | add / subtract / skip, no multiplies | integer or de-quantized fp multiplies |
| How the model gets there | trained (or fine-tuned) *with* the quantizer in the loop | rounding a finished fp model, plus calibration |
| Zero weights | ~30–40% exactly zero (measured: 31% at init, see results) | rare |

A dense model with a 4-bit QLoRA adapter is **not** a ternary model, and TriForge never calls it one.
Even TriForge's ternary-LoRA mode adds the adapter delta *before* ternarization and merges it at export,
so the exported model has ternary weights only.

## 2. Model layout

`triforge.architectures.TriForgeLM` is one LLaMA-style decoder that covers two configurations:

| | Small from-scratch models (Phase A) | BitNet b1.58 2B4T layout (Phase C/D) |
| --- | --- | --- |
| Norm before each BitLinear | RMSNorm *inside* BitLinear (paper recipe) | explicit `input_norm`, `post_attn_norm`, plus sub-layer norms before `o_proj` and `down_proj` (subln) |
| MLP | SwiGLU | ReLU²-GLU: `down(subln(relu(gate)^2 · up))` |
| Attention | MHA | GQA, 20 query / 5 KV heads |
| Position | RoPE (rotate-half, HF convention) | RoPE θ = 500 000 |
| Embedding | untied | tied, 128 256 tokens (LLaMA-3 tokenizer) |
| Size | 3.4M (validation), configs up to any size | d=2560, 30 layers, d_ff=6912: 2.41B params |

Extras used for memory efficiency: KV cache for generation, per-block gradient checkpointing,
chunked cross-entropy (`runtime.loss_chunk`) so the 128k-vocabulary fp32 logits never exist all at
once, and `PackedTernaryLinear` blocks that store only 2-bit codes.

## 3. Packed storage and kernels

`triforge.quantization.packing` stores codes losslessly as

* **2-bit**: 4 codes per byte (`0→00, +1→01, −1→10`). Easy to unpack with shifts, so it's the compute format.
* **base-3**: 5 codes per byte (3⁵ = 243 ≤ 256), i.e. 1.6 bits/weight. It's a storage format.

Neither reaches log₂3 = 1.585 bits exactly, and TriForge does not claim it does.

`triforge.quantization.kernels` has four matmul backends with one interface, all tested against each other:

* `reference`: `F.linear(x, T.to(dtype)) · gamma`, CPU/GPU, the correctness oracle (default)
* `int8`: real W1.58A8 integer arithmetic (`torch._int_mm`, int32 accumulation)
* `addonly`: multiplication-free index-add formulation, to show the arithmetic (slow in PyTorch)
* `triton`: fused 2-bit unpack + dot product on CUDA (tested on CPU via `TRITON_INTERPRET=1`)

Smaller storage does **not** automatically mean faster inference. Dense bf16 GEMMs are extremely well
optimised, and ternary only wins when a layer is memory-bandwidth bound (batch-1 decoding) *and* the
kernel is good. Measured numbers are in [RESEARCH.md](RESEARCH.md) and `results/benchmarks/`. For real
speedups on CPU, use [bitnet.cpp](https://github.com/microsoft/BitNet) with the GGUF export of 2B4T.

## 4. Package map

```
src/triforge/
  quantization/   ternary.py (quantizers, STE) · bitlinear.py (BitLinear, PackedTernaryLinear)
                  packing.py · kernels.py · convert.py (dense→ternary, reports, flip rate)
  architectures/  config.py (TriForgeConfig, analytic param counts) · transformer.py (TriForgeLM)
  training/       config.py (YAML inheritance) · build.py (model sources/strategies) · trainer.py
                  checkpoint.py (atomic, SHA-256, RNG + sampler state) · optim.py (AdamW, 8-bit, CPU offload)
  distillation/   losses.py (CE + temperature-scaled KL, chunked)
  datasets/       registry.py (licences) · pipeline.py · quality.py (dedup, decontamination) ·
                  synthetic.py (verified tasks) · tokenizer.py (BPE/HF, templates) · loader.py (memmap)
  evaluation/     benchmarks.py · runner.py · answers.py · code_exec.py · efficiency.py
  inference/      export.py · engine.py · router.py · cli.py · api.py
  huggingface/    bitnet_weights.py (2B4T loader / HF export) · hf_models.py (dense HF + ternarize) · hub.py
  hardware/       profiler.py · memory.py (training / inference memory estimator)
```

The original phase-0 code (`ternary/`, `train.py`, `compare.py` at the repo root) is kept unchanged
and still passes its tests. TriForge's `BitLinear` generalises it (optional input norm, LoRA,
progressive mix, packing).

# TernaryAI

A small language model with **ternary weights** (every weight is -1, 0 or +1), trained from scratch,
next to an identical full-precision baseline so the two can be compared directly.

The ternary layer follows **BitNet b1.58** ([Ma et al., 2024](https://arxiv.org/abs/2402.17764)):

- **Weights:** absmean quantization, `W_q = clip(round(W / mean|W|), -1, 1)`, one fp scale per matrix.
- **Activations:** per-token absmax 8-bit quantization, with an RMSNorm in front of each `BitLinear`.
- **Training:** a full-precision latent weight is kept and quantized on every forward pass;
  gradients pass through the rounding with the straight-through estimator.
- Embeddings and the output head stay full precision, as in the paper. Everything inside the
  transformer blocks (attention and SwiGLU MLP projections, ~99% of parameters) is ternary.

## Layout

| File | What it is |
| --- | --- |
| `ternary/bitlinear.py` | `BitLinear`, weight / activation quantizers |
| `ternary/model.py` | LLaMA-style decoder (RMSNorm, RoPE, SwiGLU); `linear="ternary"` or `"fp"` |
| `data.py` | Character-level TinyShakespeare (downloaded on first run) |
| `train.py` | Training loop, writes `runs/<name>/{ckpt.pt,metrics.json,sample.txt}` |
| `compare.py` | Val loss / perplexity, plus storage with weights packed at 2 bits and 1.58 bits; writes `runs/comparison.json` |
| `tests/` | Quantizer and model sanity tests (`python -m pytest tests`) |
| `runs/` | Metrics, samples and comparison from the runs below (checkpoints are not committed) |

## Quick start

```bash
pip install -r requirements.txt
python train.py --linear ternary
python train.py --linear fp
python compare.py runs/ternary runs/fp
```

`compare.py` loads each run's `ckpt.pt`, which is git-ignored, so train both runs before comparing.

Defaults are CPU-sized (4 layers, d=256, context 128, 2000 steps of batch 32).
On a GPU, try `--n_layer 6 --n_head 6 --n_embd 384 --block_size 256 --max_iters 5000`.

## First results (CPU, 2 threads per run)

Same architecture, data, seed and number of steps (2000 × 32 × 128 ≈ 8M tokens). Peak LR 3e-3 for
ternary, 1e-3 for fp (BitNet trains best with a larger LR).

| Run | Params | Val loss | Val ppl (per char) | Train time | Weight storage |
| --- | --- | --- | --- | --- | --- |
| ternary | 3.25M | 1.500 | 4.48 | 44 min | **867 KB** (2-bit packed) / 710 KB at 1.58 bits |
| fp | 3.25M | 1.476 | 4.38 | 32 min | 6,342 KB (fp16) |

What this shows so far:

- **Quality:** ternary is within ~1.6% of the full-precision loss at this tiny scale,
  and the gap was still shrinking at the end of training (it was 0.056 at step 1500, 0.024 at 2000).
- **Memory:** ternary weights are **~7.3× smaller** than fp16 (~9× at the 1.58-bit bound). Of the 867 KB, ~784 KB is the
  packed ternary weights and ~83 KB is the fp16 embedding, head and norms (~1.3% of parameters).
- **Sparsity:** ~33% of the ternary weights are exactly 0, so they can be skipped entirely by a custom kernel.
- **Generalization:** ternary had a smaller train/val gap (0.27 vs 0.35), consistent with quantization acting as a regularizer.
- **Speed (the catch):** training is *slower*, because the latent fp weights are kept and the quantizers
  add work, and here inference is also not faster because PyTorch still does fp matmuls on the
  de-quantized values. The speed and energy win of ternary only appears with kernels that replace
  multiplies with additions (e.g. [bitnet.cpp](https://github.com/microsoft/BitNet)) or with custom hardware.

These are single-seed runs on a toy dataset; treat them as a starting point, not a conclusion.

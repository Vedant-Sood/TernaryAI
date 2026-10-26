# Research findings: ternary vs binary-precision (fp / INT8 / INT4) language models

Labels used in this document: **[measured]** = produced in this repo, with the file it comes from;
**[literature]** = published result, with its source; **[estimate]** = computed from formulas, not
run; **[not run]** = planned, no result yet.

## 1. Measured results at small scale (Phase A, CPU)

Setup **[measured]**: 3.35M-parameter TriForge models (d192, 4 layers, shared 4k BPE), 1500 steps on CPU, one
seed, ternary (W1.58A8) vs an fp twin with identical data and schedule. Files: `results/<specialist>-cpu-{ternary,fp}/`.

| Specialist | Val loss ternary / fp | Synthetic held-out ternary / fp | n |
| --- | --- | --- | --- |
| TriMath | 0.3281 / 0.3130 | 20.8% / 28.3% exact match | 361 |
| TriReason | 0.3565 / 0.2762 | 12.3% / 19.4% exact match | 252 |
| TriCode | 3.223 / 3.105 | 25.8% / 25.2% (output prediction) | 329 |

Per task, ternary / fp (% exact match):

* TriMath: arithmetic 0/2, derivative 100/100, fractions 9/28, gcd 27/64, lcm 31/31, linear 21/27, word 9/13.
* TriReason: circuits 12/38, energy 0/2, kinematics 0/16, logic 50/50, momentum 3/15, syllogism 100/100,
  units 59/64, waves 16/22, work_power 0/4.

Public benchmarks on the ternary models: GSM8K 2/100, HumanEval 0/50, MBPP 0/50. At 3M parameters these
are floor results, as expected, and only show the harness runs end to end. ARC-Challenge was **[not run]**
(the HF dataset endpoint is blocked from the build container).

What it shows:
* Ternary trails fp on every reasoning-style task that needs multi-step arithmetic (fractions, gcd, circuits,
  kinematics). It matches fp on pattern tasks (derivative, syllogism, logic). TriCode is a tie.
* Training cost: ternary took 653–772 s vs 452–485 s for fp (TriMath's ternary time is inflated because tests
  ran at the same time); median throughput ~9.5k vs ~13.5k tokens/s, peak RSS 1.55 vs 1.13 GiB. The
  quantizer adds work and memory during training, as the literature predicts.
* Weights: 52.8% of parameters are ternary (embeddings dominate at this size); 32–33% of ternary weights are
  exactly zero. About 58% of codes changed over training from scratch.

## 2. Dense → ternary conversion (Phase B)

Setup **[measured]**: start from the trained fp TriMath model above, convert every linear layer to ternary,
then recover. Files: `results/trimath-cpu-{ptq,convert-*}/`.

| Method | Val loss | Synthetic EM (n=361) | Code flip rate | Train time |
| --- | --- | --- | --- | --- |
| fp source model | 0.3130 | 28.3% | – | – |
| PTQ (round, no training) | 3.4076 (ppl 30.2) | 0.8% | – | 0 |
| QAT, 300 steps | 0.3212 | 24.9% | 0.161 | 132 s |
| QAT + distillation from fp | 0.3150 | 26.0% | 0.151 | 170 s |
| Progressive ternarization + distillation | 0.3124 | 24.9% | 0.126 | 162 s |
| ternary from scratch (Phase A) | 0.3281 | 20.8% | 0.58 | 772 s |

What it shows:
* Plain rounding destroys the model (loss ×11). A short QAT run recovers most of it, and distillation from
  the fp parent helps a bit more. Recovered models beat ternary-from-scratch, but they also used more
  total compute (1500 fp steps + 300), so this isn't a like-for-like win.
* Progressive ternarization reached the lowest loss with the fewest code flips, but no better accuracy.
  Its intermediate evals use mixed fp/ternary weights; only the final number is fully ternary.
* The 360M version (`configs/phase_b/smollm2_360m_ternary_distill.yaml`) is **[not run]**; it needs the GPU.

## 3. Storage, memory and speed

**Storage [measured]** (exported files, TriMath CPU model): ternary 3,613,416 B vs fp16 6,691,848 B. Only 53% of the
small model is ternary, so the saving is 1.85×. For one 6912×2560 matrix the codes take 4.4 MB packed 2-bit vs
17.7 MB int8 vs 70.8 MB fp32 (16× smaller than fp32, 8× than fp16).

**2B memory [estimate]** (`scripts/profile_hardware.py`, BitNet 2B4T layout, 2.41B parameters, 2.08B ternary):

| Use | Estimate |
| --- | --- |
| Inference, packed 2-bit weights | 1.64 GiB |
| Inference, int8 code cache | 3.59 GiB |
| Inference, bf16 | 5.04 GiB |
| Partial QAT, last 8 blocks, 8-bit AdamW | 5.55 GiB GPU |
| LoRA rank 32 | 6.10 GiB GPU |
| Long preset, last 12 blocks, optimizer on CPU | 5.52 GiB GPU + 9.3 GiB RAM |
| Full QAT | 38.9 GiB (does not fit 8 GB) |

**Kernel speed [measured]** on CPU fp32, 4 threads, 6912×2560 (`results/benchmarks/kernels_6912x2560_cpu.json`):

| Tokens | Dense matmul | Ternary reference | Ternary int8 |
| --- | --- | --- | --- |
| 1 | 0.91 ms | 14.63 ms | 29.44 ms |
| 16 | 5.06 ms | 17.15 ms | 18.20 ms |
| 128 | 12.17 ms | 27.78 ms | 36.19 ms |

**End-to-end speed [measured]** (TriMath CPU export, batch 1, 128-token prompt, 64 new tokens): ternary prefill
4,779 tok/s and decode 295 tok/s vs fp 17,498 and 465; peak 0.6 GiB for both.

What it shows: in PyTorch, ternary is **slower** than dense (2–16× per matmul, 1.6× in decoding), because it
unpacks or casts codes and then runs an ordinary matmul. The storage saving is real; a speed gain needs
dedicated kernels (bitnet.cpp on CPU, the Triton 2-bit kernel on GPU). GPU numbers are **[not run]**.

## 4. What the literature says

* **Quality at scale [literature].** BitNet b1.58 matches a full-precision LLaMA in perplexity and zero-shot
  accuracy from about 3B parameters when both are trained on the same 100B tokens
  ([Ma et al. 2024](https://arxiv.org/abs/2402.17764)). Below that, the ternary model trails. BitNet b1.58
  2B4T (4T tokens) is competitive with fp models of 1–2B on the averages its authors report
  ([2504.12285](https://arxiv.org/abs/2504.12285)): e.g. GSM8K 58.4, MATH-500 43.4, HumanEval+ 38.4,
  ARC-Challenge 49.9, versus Qwen2.5-1.5B-Instruct 56.8 / 53.0 / 50.6 / 46.7.
* **Training from scratch is the expensive part [literature].** Those results need billions to trillions of
  tokens *with the quantizer in the loop*. Ternarizing a finished dense model (PTQ) loses most of its
  ability, and QAT recovery needs a substantial fraction of the pre-training budget. That is why TriForge
  fine-tunes a natively ternary model instead.
* **Hardware [literature].** No mainstream GPU has ternary arithmetic units. INT8 tensor cores exist,
  INT4 on recent NVIDIA parts. bitnet.cpp's CPU kernels (lookup tables / packed adds) report 1.4–6×
  CPU speedups over llama.cpp fp16 for BitNet models, mostly by cutting memory traffic
  ([microsoft/BitNet](https://github.com/microsoft/BitNet)). On GPUs the gain is limited to memory-bound
  decoding with custom kernels.

## 5. Ternary vs FP16 / INT8 / INT4: summary

| Aspect | FP16/BF16 | INT8 PTQ | INT4 PTQ (GPTQ/AWQ/NF4) | Ternary (BitNet b1.58) |
| --- | --- | --- | --- | --- |
| Bits per weight (storage) | 16 | 8 (+scales) | ~4.1–4.5 with group scales | 2 packed here (1.6 base-3; 1.58 bound) |
| How you get it | train | round a trained model | round + calibrate | **must train or fine-tune with QAT** |
| Quality loss vs FP16 | — | ~none | small at ≥7B, larger for small models | large if PTQ; small only when trained ternary at scale |
| Training memory | baseline | same as FP16 | same as FP16 (QLoRA: adapters only) | **not lower**: latent weights + optimizer, plus quantizer overhead |
| Inference arithmetic | FP multiply-add | INT8 MAC (tensor cores) | dequant + FP MAC | add/subtract/skip, if a kernel exploits it |
| Commodity kernel support | everywhere | broad | broad | bitnet.cpp (CPU), research GPU kernels |
| Exact-zero sparsity | ~0% | ~0% | ~0% | ~30–40% (measured 32–33% after training) |

## 6. Limitations of this study

* Phase A models are 3.4M parameters trained on ~6M tokens of templated data. They validate the
  implementation and give a controlled ternary vs fp comparison. They say little about 2B behaviour.
* Single seed per configuration (as in phase 0). Differences under ~0.02 nats are within run-to-run noise
  and shouldn't be read as real.
* CPU timings were taken on a shared 4-core container while other jobs ran. Treat ratios as rough.
* The 2B specialists, Phase B at 360M, and all GPU numbers are **[not run]** until the RTX 4060 runs them.

## 7. Future scaling

1. Run `bitnet2b_partial` for each specialist on the RTX 4060 and replace the training-time estimate
   with measured tokens/s. Then compare against the untouched 2B4T base on the same harness.
2. Rotate the trainable block window (LISA-style) to reach all 30 blocks within 8 GB.
3. Export to GGUF and benchmark with bitnet.cpp for real ternary speedups on the i7-13620H.
4. Multiple seeds for the small-scale comparison; sweep model size (3M → 30M → 125M) to see where
   the ternary-vs-fp gap closes in this setup.

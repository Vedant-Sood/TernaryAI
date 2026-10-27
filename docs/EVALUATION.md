# Benchmark methodology

`scripts/evaluate.py` (quality) and `scripts/benchmark.py` (efficiency) write JSON summaries plus
per-example JSONL to `results/`. Each summary records the model status (`trained`, `untrained`,
`converted-not-finetuned`, `base model`, `dense baseline`), device, decoding settings, the number of
examples `n` and the `--limit` used. A benchmark that could not run is stored as
`{"status": "not executed", "error": ...}` and is never filled with a number.

## Quality

| Specialist | Benchmark | Protocol | Metric |
| --- | --- | --- | --- |
| all | `val_loss` | held-out split of the specialist's own data, response tokens only (loss mask), 20 batches × 8 × 256 | loss, perplexity |
| all | `synthetic` | held-out verified problems from disjoint seeds, greedy decoding, `Answer:` extraction | exact match, reported per task |
| TriCode | HumanEval (164) | prompt = official function stub; generated code + official `check()`; sandboxed subprocess, 10 s timeout | pass@1, greedy |
| TriCode | MBPP test (500, task_ids 11–510) | problem text + the 3 asserts shown in the prompt (the original MBPP prompting), asserts executed | pass@1, greedy |
| TriMath | GSM8K test (1319) | zero-shot, chain of thought, final answer from `Answer:` / `####` / `\boxed{}` / last number, numeric equality | exact match |
| TriMath | MATH-500 | zero-shot; `\boxed{}` extraction; numeric equality, else normalised string equality | exact match (lower bound: no symbolic equivalence check) |
| TriReason | ARC-Challenge test (1172) | log-likelihood of each choice given the question, mean over choice tokens (lm-eval-harness `acc_norm`-style) | accuracy |

Comparisons required for every specialist:

1. **Original pretrained model** (`--bitnet microsoft/bitnet-b1.58-2B-4T-bf16`) on the same prompts.
2. **Dense / conventionally quantized baseline**: a dense model of similar size (e.g.
   `--hf Qwen/Qwen2.5-1.5B-Instruct`, `--hf HuggingFaceTB/SmolLM2-1.7B-Instruct`) evaluated with the same
   harness. At the small scale, every ternary run has an fp twin trained with identical data, steps and
   seed (`cpu_validation_fp.yaml`).

Our prompts are close to but not identical with the official harnesses (lm-evaluation-harness,
evalplus). Numbers are comparable *within this repo*; compare with published numbers only with care.
Use `--limit N` for quick checks, and report N.

## Efficiency

| Metric | How it is obtained |
| --- | --- |
| Total parameters, ternary fraction, zero fraction | `ternary_report()` on the loaded model (exact counts) |
| Disk size | size of `model.safetensors` in the export (measured) |
| Theoretical minimum size | log₂3 bits per ternary weight + 16 bits per other weight (labelled *theoretical*) |
| Inference VRAM / RAM | `torch.cuda.max_memory_allocated` / process RSS during `measure_throughput` |
| Prompt processing (prefill) | tokens/s of one forward pass over a 128-token prompt, after warm-up |
| Token generation (decode) | tokens/s of greedy decoding with KV cache, batch 1 |
| Training steps/s, tokens/s, peak memory, total time | logged in `metrics.jsonl` by the trainer |
| Matmul kernels | `benchmark.py --kernels`: dense vs reference / int8 / add-only / Triton on one layer shape |

Timing caveat: on a shared machine, other processes affect throughput. Results record the thread
count and device. Compare runs measured under the same conditions only.

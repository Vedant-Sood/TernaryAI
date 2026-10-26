# TriForge engineering report (2026-10-26)

Labels as in [RESEARCH.md](RESEARCH.md): **measured**, **estimate**, **not run**.

## 1. What was implemented

* **Ternary core** (`src/triforge/quantization/`): absmean ternarization with STE, 8-bit per-token activations,
  `BitLinear` with ternary-preserving LoRA and progressive ternarization, `PackedTernaryLinear` for frozen layers,
  2-bit and base-3 packing, and four matmul backends (reference, int8 `torch._int_mm`, add-only, Triton 2-bit).
* **Model** (`architectures/`): a GQA/RoPE transformer with ReLU²-GLU or SwiGLU, subln, KV-cache generation,
  and chunked LM loss. It loads BitNet b1.58 2B4T weights and matches `transformers`' BitNet output to within 2e-4.
* **Training** (`training/`, `distillation/`): config inheritance, scratch, BitNet, dense-to-ternary and LoRA
  builds, partial freezing, 8-bit / CPU-offloaded AdamW, KD loss, WSD/cosine LR, seq-len schedule, atomic
  SHA-256-checked checkpoints with exact resume, and an OOM report that never shrinks the model.
* **Data** (`datasets/`): a source registry with licences, exact and MinHash dedup, 13-gram decontamination,
  verified synthetic generators for all three domains, BPE/HF tokenizers and versioned manifests.
* **Evaluation** (`evaluation/`): exact match, sandboxed pass@1, multiple-choice log-likelihood and efficiency metrics.
  Results only come from runs.
* **Deployment** (`inference/`, `huggingface/`): export format, engine, router, the `triforge` CLI (chat, generate,
  serve), a FastAPI server, private-by-default HF upload/download, and model cards rendered from measured results.
* **Hardware** (`hardware/`): memory estimator and profiler with an explicit `DoesNotFit`; `check_env.py` catches
  the Anaconda OMP #15 clash and CPU-only torch.
* 19 scripts, 30+ YAML presets, 84 tests, and docs (ARCHITECTURE, INSTALL, DATA, TRAINING, EVALUATION, INFERENCE,
  HUGGINGFACE, RESEARCH, model cards).

## 2. What was trained

| Model | Params | Steps | Where |
| --- | --- | --- | --- |
| TriCode / TriMath / TriReason, ternary from scratch | 3.35M | 1500 | CPU container |
| The same three, fp twins | 3.34M | 1500 | CPU container |
| TriMath fp→ternary: PTQ, QAT, QAT+KD, progressive+KD | 3.35M | 0 / 300 / 300 / 150 | CPU container |

**No 2B specialist has been trained.** They need the RTX 4060 and the BitNet download (huggingface.co is blocked
from the container).

## 3. What was tested

84 tests pass (`python -m pytest -q`). They cover quantization and packing round trips, kernel agreement,
HF BitNet parity, partial freeze, export/reload in a fresh process, exact resume and checkpoint integrity, data
dedup and decontamination, router, answer matching, sandboxed code execution, the API and the profiler.

## 4. Model sizes

* CPU models: 3,350,720 parameters, 52.8% ternary. Export is 3.61 MB vs 6.69 MB fp16 (**measured**).
* 2B specialists: 2,412,820,480 parameters, 2,084,044,800 ternary (86.4%). Packed inference needs ~1.64 GiB
  (**estimate**).

## 5. Measured results (CPU, single seed)

* Ternary vs fp val loss: TriMath 0.328 vs 0.313, TriReason 0.357 vs 0.276, TriCode 3.22 vs 3.11.
* Ternary vs fp held-out exact match: TriMath 20.8% vs 28.3%, TriReason 12.3% vs 19.4%, TriCode 25.8% vs 25.2%.
* Converting fp TriMath to ternary: PTQ gave loss 3.41 and 0.8% EM. 300 QAT steps recovered it to 0.321 / 24.9%,
  and adding distillation gave 0.315 / 26.0%.
* Speed: ternary training ran ~30% slower. In PyTorch, ternary decode was 295 vs 465 tok/s, and per-matmul
  2–16× slower than dense. Storage per matrix was 8× smaller than fp16.
* GSM8K 2/100, HumanEval 0/50 and MBPP 0/50 are floor scores at 3M parameters.

Details and caveats are in [RESEARCH.md](RESEARCH.md).

## 6. Hugging Face

Nothing has been published. The planned repos are `VedJSood/{TriCode,TriMath,TriReason}-2B-ternary`, all **private**.
Uploads need explicit approval, and `upload_hf.py` refuses to push to a public repo unless `--public` is given.

## 7. Not done yet, and why

| Item | Reason |
| --- | --- |
| 2B partial-QAT runs for the three specialists | needs the RTX 4060 and the HF download |
| BitNet 2B4T base-model baseline on the same harness | same |
| Phase B at 360M (SmolLM2) | needs a GPU |
| GPU throughput, Triton kernel timing | no GPU in the container |
| ARC-Challenge, MATH-500 | the HF datasets endpoint is blocked from the container |
| bitnet.cpp / GGUF export | not implemented; the next step for real CPU speedups |
| Multi-seed runs | time; single-seed differences under ~0.02 nats are noise |

## 8. Exact commands

```bash
# setup (WSL2 Ubuntu)
bash scripts/install_wsl.sh && source .venv/bin/activate
python scripts/check_env.py && python -m pytest -q
hf auth login && python scripts/check_hf_auth.py

# reproduce the CPU results
bash scripts/run_cpu_validation.sh      # data + 6 training runs + exports
bash scripts/eval_cpu_validation.sh     # val loss, synthetic, GSM8K/HumanEval/MBPP
bash scripts/run_phase_b_cpu.sh         # PTQ / QAT / QAT+KD / progressive
python scripts/benchmark.py --kernels --shape 6912 2560

# a 2B specialist on the RTX 4060 (repeat for tricode, trireason)
python scripts/profile_hardware.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/prepare_data.py     --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/train.py            --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto
python scripts/export.py   --run checkpoints/trimath/bitnet2b_partial --out checkpoints/exports/trimath-2b
python scripts/evaluate.py --export checkpoints/exports/trimath-2b --specialist trimath --data data/trimath/v1-bitnet
python scripts/make_model_cards.py

# inference
triforge chat -m auto
triforge serve --port 8765

# private upload, only after approval
python scripts/upload_hf.py --export checkpoints/exports/trimath-2b --repo VedJSood/TriMath-2B-ternary --card docs/model_cards/TriMath.md
```

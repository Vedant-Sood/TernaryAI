# Training strategy and commands

## 1. Which approach, and why

| Approach | What it is | Evidence | Fits 8 GB for ~2B? | Verdict |
| --- | --- | --- | --- | --- |
| Ternary **from scratch** | random init, STE from step 0 | BitNet b1.58 matches fp LLaMA at ≥3B *given equal tokens* (Ma et al. 2024); 2B4T used **4T tokens** | memory yes (partial), **compute no**: 4T tokens on a laptop is centuries | Phase A only (3.4M-param validation models) |
| **Post-training ternarization** of a dense model | round a finished fp model to {−1,0,+1} | ternary PTQ of dense LLMs collapses; even 2-bit PTQ needs heavy machinery | yes | measured as a baseline in Phase B; not a deliverable |
| **QAT fine-tuning** of a dense model | ternarize, then keep training with STE | recovers part of the loss; needs billions of tokens to approach the original | 360M yes, 2B only partially | Phase B experiment (SmolLM2-360M) |
| **Distillation** into a ternary student | QAT + KL to the dense teacher's logits | standard way to speed up recovery | teacher + student must share 8 GB: OK at 360M | Phase B experiment |
| **Progressive ternarization** | blend `(1−λ)W + λQ(W)`, λ: 0→1 | softens the PTQ shock | same as QAT | option (`optim.progressive_ternarization_steps`) |
| **QAT fine-tuning of a *native* ternary model** | start from BitNet b1.58 2B4T (already W1.58A8, MIT) and specialise it | it never leaves the ternary regime it was pre-trained in | **yes**, with frozen packed blocks | **Primary path to the 2B specialists** |
| Ternary-preserving **LoRA** | low-rank delta added to the latent weight *before* ternarization | cheap, but a small delta flips few codes | yes (seq ≤ 1024) | alternative preset; flip rate is measured |
| QLoRA on a 4-bit dense model | dense 4-bit base + bf16 adapter | — | yes | **rejected**: not a ternary model |
| CPU offloading | optimizer + fp32 master in RAM | standard (ZeRO-Offload) | yes | used by the long-running preset |

**Decision.** The deployable TriCode / TriMath / TriReason are **specialised fine-tunes of BitNet b1.58
2B4T** (2.41B parameters, 2.08B ternary). Every forward weight stays ternary during and after
fine-tuning. Training from scratch is used only to validate the implementation (Phase A) and to
compare ternary against fp at equal settings. Dense→ternary conversion (Phase B) is measured to
quantify how much is lost and recovered, because the user asked for it and it informs the research
question. It is not the main route.

## 2. Memory

Low-bit inference memory says little about training memory. Estimates come from `scripts/profile_hardware.py`
(`triforge/hardware/memory.py`). They are **estimates, not measurements**, for an 8 GB card with ~6.7 GiB usable:

| Preset (BitNet 2B4T) | Trainable | GPU estimate | Fits 8 GB? |
| --- | --- | --- | --- |
| `bitnet2b_partial_rtx4060` (last 8 of 30 blocks, 8-bit AdamW, seq 2048, mb 1) | 556M | **5.6 GiB** (1.0 weights + 1.0 grads + 1.0 optim + 1.0 frozen packed blocks & embedding + 0.5 activations/logits + 0.9 overhead) | yes |
| `bitnet2b_lora_rtx4060` (rank 32, all blocks, seq 1024) | ~40M | **6.1 GiB** (latent bf16 weights of all blocks must stay resident) | yes, tight |
| `bitnet2b_long_rtx4060` (last 12 blocks, optimizer in RAM) | 834M | **5.5 GiB** GPU + ~9.3 GiB RAM | yes |
| `bitnet2b_full_large` (all blocks, AdamW, seq 4096, mb 8) | 2.08B | **~39 GiB** | no: ≥40 GB GPU |
| any full fine-tune with fp32 AdamW at mb 1 | 2.4B | > 20 GiB | no |

Inference (estimate): packed 2-bit weights ~1.1 GiB + KV cache; with int8-cached codes (faster in
the PyTorch path) ~3.0 GiB; bf16 latents 4.5 GiB.

Techniques used: bf16 autocast, bf16 parameter storage, gradient accumulation, per-block gradient
checkpointing, sequence-length warm-up schedule, chunked loss over the 128k vocabulary, 8-bit AdamW
(bitsandbytes) or CPU-offloaded AdamW, frozen blocks stored as 2-bit codes (no latent weight, no
grad, no optimizer state), memory-mapped token shards, PyTorch SDPA (flash / mem-efficient attention).

## 3. Presets

Each specialist directory under `configs/` has the same presets:

| File | Purpose | Hardware |
| --- | --- | --- |
| `smoke.yaml` | architecture smoke test, 30 steps, whole pipeline | any CPU, ~10 s |
| `cpu_validation.yaml` / `_fp.yaml` | short experimental run, ternary vs dense, 1500 steps | CPU, ~12 min each |
| `bitnet2b_quick_rtx4060.yaml` | ~30-minute pipeline check at full 2B size (150 steps, seq 512, hard `runtime.max_train_minutes: 30` cap) | RTX 4060 8 GB |
| `bitnet2b_partial_rtx4060.yaml` | full local fine-tuning experiment | RTX 4060 8 GB |
| `bitnet2b_lora_rtx4060.yaml` | ternary-LoRA alternative | RTX 4060 8 GB |
| `bitnet2b_long_rtx4060.yaml` | long-running session (days, resumable) | RTX 4060 + 32 GB RAM |
| `bitnet2b_full_large.yaml` | larger hardware, all blocks trainable | ≥ 40 GB GPU |
| `bitnet2b_{quick,partial}_t4.yaml` | same runs on a free Colab or Kaggle T4: fp16 + loss scaling, fp32 trainable weights, resumed across sessions ([CLOUD_GPU.md](CLOUD_GPU.md)) | T4 16 GB |
| `bitnet2b_{quick,partial}_l4.yaml` | same runs on a 24 GB bf16 GPU (L4 / A10G, e.g. Modal): fp32 trainable weights, micro-batch 2 ([CLOUD_GPU.md](CLOUD_GPU.md)) | L4 / A10G 24 GB |

Configs are split into `run / model / data / optim / runtime / eval / distill` sections with
`inherit:` for shared bases (`configs/base/`). Override anything on the command line:
`--set optim.lr=5e-6 runtime.micro_batch_size=2`.

## 4. Commands

```bash
# 0. environment and hardware
python scripts/check_env.py
python scripts/check_hf_auth.py
python scripts/profile_hardware.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml

# 1. Phase A: CPU validation, all three specialists, ternary vs fp
bash scripts/run_cpu_validation.sh

# 2. Phase B: dense -> ternary (SmolLM2-360M)
python scripts/prepare_data.py --config configs/phase_b/smollm2_360m_ternary_distill.yaml
python scripts/convert_pretrained.py --model HuggingFaceTB/SmolLM2-360M --data data/phase_b/v1-fineweb
python scripts/train.py --config configs/phase_b/smollm2_360m_ternary_distill.yaml

# 3. Phase C/D: 2B specialists (repeat for tricode / trireason)
python scripts/prepare_data.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/evaluate.py --bitnet microsoft/bitnet-b1.58-2B-4T-bf16 --specialist trimath \
    --data data/trimath/v1-bitnet --limit 200 --name bitnet2b4t-base      # baseline BEFORE training
python scripts/train.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto
python scripts/export.py --run checkpoints/trimath/bitnet2b_partial --out checkpoints/exports/trimath-2b
python scripts/validate_ternary.py --export checkpoints/exports/trimath-2b
python scripts/evaluate.py --export checkpoints/exports/trimath-2b --specialist trimath \
    --data data/trimath/v1-bitnet --limit 200

# 4. inference
triforge chat -m auto
```

## 5. Checkpoints and resuming

* `checkpoints/<specialist>/<run>/ckpt/step_XXXXXXX/` holds `model.safetensors`, `optim.pt`, `rng.pt`
  and `state.json` (step, tokens seen, data-sampler state, SHA-256 of the model file).
* Writes are atomic (temp dir + rename; `latest` pointer updated last), so a crash mid-save keeps
  the previous checkpoint usable. `keep_last` controls retention.
* Several GPUs: `torchrun --standalone --nproc_per_node=N scripts/train.py ...` (data parallel; gradients
  averaged, accumulation split across GPUs, rank 0 saves). Checkpoints are interchangeable with single-GPU runs.
* `--resume auto` continues from `latest`; the resumed run reproduces the uninterrupted run's losses
  exactly (`test_resume_reproduces_uninterrupted_run`).
* Ctrl-C saves a checkpoint before exiting (exit code 130). A CUDA OOM saves one if possible and
  writes `oom_report.json` (exit code 3). A non-finite loss saves and stops (exit code 4).
* Sync a checkpoint to a private Hub repo branch with `triforge.huggingface.hub.sync_checkpoint`.

## 6. Training-time budget for the 2B presets

Not measured yet: it needs the RTX 4060. Rough planning number: with frozen blocks, a step costs a
full 2.4B forward plus backward through 8 blocks. At an assumed 3–6 effective TFLOPS on a 60 W laptop
GPU, that is roughly 0.5–1k tokens/s. One `bitnet2b_partial` run (600 steps × 32 × 2048 ≈ 39M
tokens) would take 11–22 hours. Replace this with the measured `tokens_per_s` from `metrics.jsonl`
after the first run.

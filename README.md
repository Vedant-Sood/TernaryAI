# TernaryAI / TriForge

**TriForge** is a framework for language models whose transformer weights are **ternary**: every
weight is −1, 0 or +1, with one scale per matrix, and matmul inputs are 8-bit (W1.58A8, following
[BitNet b1.58](https://arxiv.org/abs/2402.17764)). It has three specialist pipelines:

| Specialist | Domain | 2B config | Small validated model |
| --- | --- | --- | --- |
| **TriCode** | programming, algorithms, debugging, tests | `configs/tricode/bitnet2b_*.yaml` | `configs/tricode/cpu_validation.yaml` |
| **TriMath** | algebra, calculus, number theory, proofs, word problems | `configs/trimath/bitnet2b_*.yaml` | `configs/trimath/cpu_validation.yaml` |
| **TriReason** | physics, engineering, logic, quantitative reasoning | `configs/trireason/bitnet2b_*.yaml` | `configs/trireason/cpu_validation.yaml` |

> **Status (2026-10-26).** The framework, the 84-test suite and the three CPU-scale (3.4M-parameter)
> validation models are done and measured, ternary vs dense. **The 2B specialists are not trained
> yet.** Their configs, memory plans and pipeline are ready, and they need the RTX 4060 laptop and a
> Hugging Face download. See [docs/REPORT.md](docs/REPORT.md) for what is trained and measured and what isn't.

## How the 2B specialists are built

Training a 2B ternary model from scratch took Microsoft 4 trillion tokens, which a laptop can't do.
TriForge therefore fine-tunes **[BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)**
(MIT), a model that is *natively* ternary, with quantization-aware training, so the forward weights
stay ternary throughout. To fit 8 GB of VRAM, only the last 8 of 30 blocks keep bf16 latent weights.
The other blocks are frozen as packed 2-bit codes. The estimate is 5.6 GiB
([docs/TRAINING.md](docs/TRAINING.md)). Converting *dense* models to ternary is measured as a
separate experiment (Phase B).

## Quick start

```bash
# WSL2 Ubuntu (recommended; Windows-native instructions in docs/INSTALL.md)
git clone https://github.com/Vedant-Sood/TernaryAI.git && cd TernaryAI
bash scripts/install_wsl.sh && source .venv/bin/activate
python scripts/check_env.py                    # GPU / CUDA / packages; flags the Anaconda OMP #15 clash
python -m pytest -q                            # 84 tests

# Phase A on CPU, ~1.5 h: data -> train ternary and fp twins -> export -> evaluate
bash scripts/run_cpu_validation.sh
bash scripts/eval_cpu_validation.sh

# 2B specialist on the RTX 4060 (TriMath shown; same for tricode / trireason)
hf auth login
python scripts/profile_hardware.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/prepare_data.py   --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/train.py          --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto
python scripts/export.py --run checkpoints/trimath/bitnet2b_partial --out checkpoints/exports/trimath-2b
python scripts/evaluate.py --export checkpoints/exports/trimath-2b --specialist trimath --data data/trimath/v1-bitnet

# local inference (CLI and HTTP API), offline once exports are installed
triforge chat -m auto
triforge serve --port 8765
```

## Documentation

| Doc | Contents |
| --- | --- |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | ternary weights, STE, what stays high precision, ternary vs INT4/INT8, packing, kernels, code map |
| [INSTALL](docs/INSTALL.md) | Windows / WSL2 / CUDA setup, OMP Error #15, troubleshooting |
| [TRAINING](docs/TRAINING.md) | strategy comparison and choice, 8 GB memory plan, presets, commands, resuming |
| [CLOUD_GPU](docs/CLOUD_GPU.md) | training the 2B specialists on Modal, Colab or Kaggle GPUs, resumed across sessions |
| [DATA](docs/DATA.md) | datasets, licences, dedup and decontamination, mixtures |
| [EVALUATION](docs/EVALUATION.md) | benchmark protocols and efficiency metrics |
| [INFERENCE](docs/INFERENCE.md) | CLI, local API, router |
| [HUGGINGFACE](docs/HUGGINGFACE.md) | base-model selection, auth, private repos, upload / download |
| [RESEARCH](docs/RESEARCH.md) | measured findings: ternary vs fp, conversion, sparsity, speed, limits |
| [REPORT](docs/REPORT.md) | engineering report: done, trained, measured, remaining |
| [Model cards](docs/model_cards/) | TriCode, TriMath, TriReason |

## Repository layout

```
configs/{base,tricode,trimath,trireason,phase_b}/   YAML presets (inherit + override)
src/triforge/   quantization · architectures · training · distillation · datasets · evaluation ·
                inference · huggingface · hardware
scripts/        check_env, check_hf_auth, profile_hardware, prepare_data, init_model, validate_ternary,
                convert_pretrained, train, export, evaluate, benchmark, infer, serve, upload_hf, download_hf
tests/          84 tests (quantization, model, HF compatibility, data, training/resume, export, API, eval)
docs/  results/  checkpoints/ (git-ignored)  data/ (git-ignored)
ternary/ train.py compare.py data.py runs/   phase-0 TinyShakespeare experiment (kept as is)
```

## Phase 0 (kept): TinyShakespeare, ternary vs fp

The first experiment (PR #1) still runs unchanged: `pip install -r requirements.txt`, then `python train.py --linear ternary`,
`python train.py --linear fp` and `python compare.py`. With 3.25M parameters, char-level, 2000 steps on CPU:
ternary reached val loss 1.500 and fp 1.477. Ternary weights were 7.3× smaller (2-bit packed), 33% of the
ternary weights were zero, and ternary trained ~35% slower in PyTorch.

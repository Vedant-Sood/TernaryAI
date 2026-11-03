# TernaryAI / TriForge

A research framework for ternary-weight language models and three specialist
pipelines: **TriCode**, **TriMath** and **TriReason**. It explores the storage,
quality and runtime tradeoffs of weights restricted to -1, 0 and +1.

**Status:** small CPU-scale models and experiments have recorded results.
The 2B specialist configurations are provided, but **the 2B specialists have
not been trained**. The [engineering report](docs/REPORT.md) separates measured
results, estimates and work that has not run.

## What is here

| Component | Scope |
| --- | --- |
| TriCode | Coding, algorithms, debugging and tests |
| TriMath | Mathematics, proofs and word problems |
| TriReason | Physics, engineering and quantitative reasoning |
| Training framework | Ternary quantization, packed frozen layers, LoRA, distillation and resume |
| Evaluation | Quality benchmarks, storage measurements and runtime comparisons |
| Local inference | Specialist router, CLI and HTTP API for installed exports |
| Phase 0 | Original character-level TinyShakespeare comparison |

The modern framework is in src/triforge. The root train.py, compare.py,
data.py and ternary directory belong to the earlier Phase 0 experiment.

## Install

The existing [installation guide](docs/INSTALL.md) covers WSL2, native Windows,
CUDA dependencies and the OpenMP conflict. From Ubuntu/WSL2:

~~~bash
git clone https://github.com/Vedant-Sood/TernaryAI.git
cd TernaryAI
bash scripts/install_wsl.sh
source .venv/bin/activate
python scripts/check_env.py
python -m pytest -q
~~~

The install and data-preparation steps can download large dependencies or models.
The report records 84 tests from the earlier validation; this README does not
claim a new run.

## Reproduce the small experiments

~~~bash
bash scripts/run_cpu_validation.sh
bash scripts/eval_cpu_validation.sh
~~~

The first script prepares data, trains ternary/dense twins and exports them.
The second evaluates the exports and invokes benchmark datasets. These are
training and evaluation jobs, not instant demos; initial preparation and
benchmark downloads require internet access.

### Recorded CPU-scale results

From [docs/REPORT.md](docs/REPORT.md), dated 26 October 2026; single-seed results:

| Specialist | Ternary validation loss | Dense validation loss |
| --- | --- | --- |
| TriMath | 0.328 | 0.313 |
| TriReason | 0.357 | 0.276 |
| TriCode | 3.22 | 3.11 |

The specialist models have about 3.35M parameters; their recorded export sizes
were 3.61 MB versus 6.69 MB for fp16. These are different from the original
3.25M-parameter TinyShakespeare experiment. Small synthetic-domain results do
not establish the quality of a trained 2B specialist.

## 2B specialist workflow

The configs fine-tune the natively ternary BitNet b1.58 2B4T base model rather
than training a 2B model from scratch. The laptop presets freeze and pack most
blocks while retaining trainable latent weights in selected blocks.
Memory figures are plans or estimates until the selected hardware is profiled.

TriMath example, after installation and any required Hugging Face authentication:

~~~bash
python scripts/profile_hardware.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/prepare_data.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml
python scripts/train.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto
python scripts/export.py --run checkpoints/trimath/bitnet2b_partial --out checkpoints/exports/trimath-2b
python scripts/evaluate.py --export checkpoints/exports/trimath-2b --specialist trimath --data data/trimath/v1-bitnet
~~~

Use the corresponding tricode or trireason configs for the other specialists.
Read [training](docs/TRAINING.md) and [cloud GPU options](docs/CLOUD_GPU.md)
before choosing a larger preset. A configuration file is not a trained checkpoint.

## Local inference

Run from the repository root after installing or creating compatible exports
and configuring configs/models.yaml:

~~~bash
triforge models
triforge chat -m auto
triforge serve --port 8765
~~~

The API requires the serve extra. See [inference setup](docs/INFERENCE.md)
for registry paths, export formats and supported backends. Neither chat nor
serve creates missing weights.

## Original TinyShakespeare experiment

~~~bash
python -m pip install -r requirements.txt
python train.py --linear ternary
python train.py --linear fp
python compare.py
~~~

Train both runs before comparing: compare.py loads their ignored ckpt.pt files.
The original 3.25M-parameter, 2,000-step comparison reported validation losses
of 1.500 (ternary) and about 1.477 (dense), and roughly 7.3x smaller packed
weight storage than fp16. These numbers describe that experiment only.

## Documentation and layout

Start at the [documentation index](docs/README.md).

| Path | Purpose |
| --- | --- |
| [src/triforge/](src/triforge/) | Current model, quantization, training, data, inference and evaluation code |
| [configs/](configs/) | Specialist, hardware and conversion presets |
| [scripts/](scripts/) | Environment checks, data preparation, train/export/evaluate and publishing tools |
| [tests/](tests/) | Framework checks |
| [results/](results/) | Committed experiment summaries and measurements |
| [docs/](docs/) | Methods, installation, training, reports and model cards |
| [ternary/](ternary/) | Earlier Phase 0 implementation |

Packed storage savings do not automatically produce faster PyTorch execution.
The recorded reference path was slower than dense. GPU throughput, large-model
quality and specialized-kernel benefits must be measured separately.
Project code uses the [MIT license](LICENSE); datasets and base models retain
their own terms.

# TriForge documentation

[Project overview](../README.md)

| Question | Guide |
| --- | --- |
| What has actually been built, trained and measured? | [Engineering report](REPORT.md) |
| How do I set up Windows, WSL2 or CUDA? | [Installation](INSTALL.md) |
| How are ternary weights, packing and kernels implemented? | [Architecture](ARCHITECTURE.md) |
| Which training strategy and memory preset should I use? | [Training](TRAINING.md) |
| How do I use a cloud GPU? | [Cloud GPU options](CLOUD_GPU.md) |
| Where do the data and licenses come from? | [Data](DATA.md) |
| How is quality or efficiency measured? | [Evaluation](EVALUATION.md) |
| How do local exports, routing, chat and the API work? | [Inference](INFERENCE.md) |
| How are Hugging Face downloads and uploads handled? | [Hugging Face](HUGGINGFACE.md) |
| What do the experiments show, including negative results? | [Research](RESEARCH.md) |
| What is each specialist intended to do? | [TriCode](model_cards/TriCode.md), [TriMath](model_cards/TriMath.md), [TriReason](model_cards/TriReason.md) |

## Keep the experiments separate

Phase 0 uses the root train.py and compare.py for character-level
TinyShakespeare models. The later framework uses src/triforge, YAML configs
and scripts/train.py for specialist experiments. Their losses, tokenization
and model sizes are not interchangeable.

The report records CPU-scale experiments. The 2B presets are untrained, and
large-model memory estimates are not measured throughput or quality results.
Checkpoints are not supplied merely by cloning the source tree. Start with
REPORT.md before interpreting benchmark tables or planning a larger run.

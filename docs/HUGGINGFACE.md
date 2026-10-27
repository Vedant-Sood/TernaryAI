# Hugging Face integration

## What the connected account can do (checked 2026-10-26)

The Claude Hugging Face connector is authenticated as **VedJSood** (free account, no organisations).
Its OAuth scopes are `read-repos`, `contribute-repos`, `inference-api` and `jobs`, and the token
expired at 10:54 UTC that day. It was used to:

* search and inspect models and datasets (licences, configs, file lists),
* read `config.json`, `README.md` and `tokenizer_config.json` of `microsoft/bitnet-b1.58-2B-4T(-bf16)`.

The connector has **no repository-creation or file-upload operation**, and huggingface.co was blocked
from the cloud container. So nothing was uploaded, and model repositories are created and uploaded
from the laptop with the standard `huggingface_hub` SDK instead.

## Base models considered

| Model | Licence | Params | Ternary? | Role |
| --- | --- | --- | --- | --- |
| microsoft/bitnet-b1.58-2B-4T-bf16 | MIT | 2.41B | **native** W1.58A8, 4T tokens | **base for all three specialists** (bf16 master weights, meant for training) |
| microsoft/bitnet-b1.58-2B-4T | MIT | same | packed inference weights | reference for inference only |
| microsoft/bitnet-b1.58-2B-4T-gguf | MIT | same | bitnet.cpp format | fast CPU inference |
| 1bitLLM/bitnet_b1_58-large | MIT | 0.73B | native (reproduction, 100B tokens) | smaller fallback |
| HuggingFaceTB/SmolLM2-360M | Apache-2.0 | 0.36B | dense | Phase B conversion experiment + teacher |
| Qwen/Qwen2.5-0.5B / 1.5B-Instruct | Apache-2.0 | 0.49B / 1.5B | dense | dense baselines for evaluation |

## Authentication (never commit tokens)

```bash
pip install -U huggingface_hub
hf auth login                      # paste a token with write access; stored in ~/.cache/huggingface
python scripts/check_hf_auth.py    # shows user and token role, never the token
```

`HF_TOKEN` in the shell environment also works. Never put it in a file inside the repository
(`.gitignore` blocks `.env`, `*.token`, `hf_token*`).

## Repositories (private by default)

Planned names (create only after a *trained* export exists and you approve):

| Specialist | Repo |
| --- | --- |
| TriCode | `VedJSood/TriCode-2B-ternary` |
| TriMath | `VedJSood/TriMath-2B-ternary` |
| TriReason | `VedJSood/TriReason-2B-ternary` |

```bash
python scripts/upload_hf.py --export checkpoints/exports/trimath-2b \
    --repo VedJSood/TriMath-2B-ternary --card docs/model_cards/TriMath.md
```

`upload_hf.py` creates the repo **private**, refuses to upload into an existing *public* repo unless
`--public` is given, refuses exports whose status is not `trained` (override: `--allow-untrained`), and
skips optimizer states and checkpoints. Download for offline use:

```bash
python scripts/download_hf.py --repo VedJSood/TriMath-2B-ternary --out checkpoints/exports/trimath-2b
```

Checkpoint sync for long runs (pushes the latest checkpoint to a `checkpoints` branch of the private repo):

```python
from triforge.huggingface.hub import sync_checkpoint
sync_checkpoint("checkpoints/trimath/bitnet2b_long", "VedJSood/TriMath-2B-ternary", step=1200)
```

## Hub conventions used

* Weights as `safetensors`; tokenizer saved with `save_pretrained`; configs as JSON.
* `scripts/export.py --hf-bf16 DIR` additionally writes the HF **BitNet** layout (`BitNetForCausalLM`,
  `quantization_mode: online`), loadable by `transformers>=4.52` with `AutoModelForCausalLM`. The ternary
  codes are preserved exactly; per-matrix scales go through bf16 rounding (tested: ≤0.1 logit difference on a tiny model).
* Datasets are streamed (`load_dataset(..., streaming=True)`), so no source has to be downloaded completely.
* Pin `revision:` of base models and datasets in configs for reproducible provenance; manifests and
  `triforge.json` record what was used.

# Local inference

Models are registered in `configs/models.yaml` (name → export directory). Exports are self-contained
(weights, config, tokenizer, `triforge.json` with status and provenance), so once the files are on disk
nothing needs the internet. For exports with a Hugging Face tokenizer, set `HF_HUB_OFFLINE=1` to be sure.

## CLI

```bash
pip install -e .                      # provides the `triforge` command (or use python scripts/infer.py ...)
triforge models                       # registered models; installed? loaded?
triforge route "Integrate x^2 sin x"  # router recommendation + per-specialist scores
triforge generate -m auto --temperature 0 --max-new-tokens 200 "Solve 3x + 5 = 20"
triforge chat -m tricode              # interactive
triforge sysinfo                      # GPU / RAM snapshot
triforge --backend int8 chat          # pick the ternary matmul backend (reference | int8 | addonly | triton)
```

Commands inside `chat`:

| Command | Effect |
| --- | --- |
| `/model NAME` or `/model auto` | choose a specialist or let the router decide |
| `/temp 0.2`, `/max 512`, `/topk 40`, `/topp 0.9` | generation settings |
| `/stats` | last generation: prompt tokens, new tokens, time to first token, prefill and decode tokens/s, RAM/GPU use, router scores |
| `/load NAME`, `/unload NAME`, `/models` | manage memory |
| `/quit` | exit |

## Local API

```bash
triforge serve --port 8765            # binds 127.0.0.1; no authentication; don't expose it
curl -s localhost:8765/models
curl -s -X POST localhost:8765/generate -H 'content-type: application/json' \
     -d '{"prompt": "Write a Python function that checks for palindromes", "model": "auto", "temperature": 0}'
```

Endpoints: `GET /health`, `GET /models`, `POST /load`, `POST /unload`, `POST /route`, `POST /generate`,
`GET /system`. `/generate` returns `{"text", "stats"}` with the same stats as `/stats`.

## Router

`triforge.inference.router` is a deterministic, rule-based scorer (regex cues for code, maths and
science / logic, weighted). It is unit-tested (`tests/test_inference_eval.py`) and can always be
overridden (`model=` in the API, `/model` in chat). With no cues it falls back to TriReason. It
recommends; it doesn't block anything.

## Memory and speed

The default `reference` backend unpacks codes once to int8 and runs bf16 matmuls. That is reliable,
not fast. Expect speed *comparable to or slower than* a dense model of the same shape in PyTorch, with
lower weight memory. Microsoft's model card says the same about `transformers`. For fast CPU decoding of 2B4T-derived models,
convert to GGUF and use [bitnet.cpp](https://github.com/microsoft/BitNet). Measured numbers: `results/benchmarks/`.

#!/usr/bin/env bash
# Export and evaluate the Phase A CPU validation models (ternary and fp twins).
set -euo pipefail
cd "$(dirname "$0")/.."
for s in trimath trireason tricode; do
  for v in ternary fp; do
    python scripts/export.py --run checkpoints/$s/cpu_$v --out checkpoints/exports/$s-cpu-$v
    python scripts/evaluate.py --export checkpoints/exports/$s-cpu-$v --specialist $s --data data/$s/v0-cpu \
        --benchmarks val_loss synthetic
  done
done
python scripts/evaluate.py --export checkpoints/exports/trimath-cpu-ternary --specialist trimath --benchmarks gsm8k --limit 100
python scripts/evaluate.py --export checkpoints/exports/tricode-cpu-ternary --specialist tricode --benchmarks humaneval mbpp --limit 50
python scripts/evaluate.py --export checkpoints/exports/trireason-cpu-ternary --specialist trireason --benchmarks arc_challenge --limit 100

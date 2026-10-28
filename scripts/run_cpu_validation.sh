#!/usr/bin/env bash
# Phase A on CPU: train ternary and dense-fp models with identical settings for all three
# specialists, then export. Evaluation: scripts/evaluate.py (see README).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_data.py --config configs/tricode/cpu_validation.yaml configs/trimath/cpu_validation.yaml \
    configs/trireason/cpu_validation.yaml
for s in trimath trireason tricode; do
  for v in cpu_validation cpu_validation_fp; do
    python scripts/train.py --config configs/$s/$v.yaml --resume auto
  done
  python scripts/export.py --run checkpoints/$s/cpu_ternary --out checkpoints/exports/$s-cpu-ternary
  python scripts/export.py --run checkpoints/$s/cpu_fp --out checkpoints/exports/$s-cpu-fp
done

#!/usr/bin/env bash
# Phase B at CPU scale: dense TriMath validation model -> ternary. Needs checkpoints/trimath/cpu_fp.
set -euo pipefail
cd "$(dirname "$0")/.."
for c in cpu_trimath_ptq cpu_trimath_qat cpu_trimath_qat_distill cpu_trimath_qat_progressive; do
  python scripts/train.py --config configs/phase_b/$c.yaml
done
python scripts/export.py --run checkpoints/phase_b/cpu_trimath_ptq --out checkpoints/exports/trimath-cpu-ptq \
    --status converted-not-finetuned
for c in qat qat_distill progressive; do
  python scripts/export.py --run checkpoints/phase_b/cpu_trimath_$c --out checkpoints/exports/trimath-cpu-convert-$c
done
for e in ptq convert-qat convert-qat_distill convert-progressive; do
  python scripts/evaluate.py --export checkpoints/exports/trimath-cpu-$e --specialist trimath \
      --data data/trimath/v0-cpu --benchmarks val_loss synthetic
done

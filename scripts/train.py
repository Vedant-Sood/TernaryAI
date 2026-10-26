"""Train / fine-tune / distill a TriForge model from a YAML config.

    python scripts/train.py --config configs/trimath/cpu_validation.yaml
    python scripts/train.py --config configs/trimath/bitnet2b_partial_rtx4060.yaml --resume auto
    python scripts/train.py --config ... --set optim.lr=5e-6 runtime.micro_batch_size=1

Exit codes: 0 ok, 3 CUDA out of memory (diagnostics in <output_dir>/oom_report.json), 4 non-finite loss.
"""
import argparse
import sys

import _common  # noqa: F401
import torch

from triforge.training import Trainer, load_config, oom_report
from triforge.training.config import resolve_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="'auto' (newest checkpoint) or a checkpoint directory")
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.lr=1e-3")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    trainer = None
    try:
        trainer = Trainer(cfg)
        if args.resume:
            trainer.resume(args.resume)
        trainer.fit()
    except torch.OutOfMemoryError as e:
        if trainer is not None:
            try:
                trainer.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                trainer.save({"reason": "oom"})
            except Exception as save_err:  # the save itself can OOM
                print(f"could not save emergency checkpoint: {save_err}", file=sys.stderr)
        oom_report(trainer or cfg, e, resolve_path(cfg["run"]["output_dir"]))
        sys.exit(3)
    except FloatingPointError as e:
        print(f"training diverged: {e}", file=sys.stderr)
        sys.exit(4)
    except KeyboardInterrupt:
        if trainer is not None:
            trainer.save({"reason": "interrupted"})
        sys.exit(130)


if __name__ == "__main__":
    main()

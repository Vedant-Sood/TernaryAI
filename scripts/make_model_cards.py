"""Render docs/model_cards/{TriCode,TriMath,TriReason}.md from measured results in results/.

    python scripts/make_model_cards.py
    python scripts/make_model_cards.py --status-file docs/model_cards/status.yaml
"""
import argparse
import glob
import os

import _common  # noqa: F401
import yaml

from triforge.huggingface.model_card import SPEC, render
from triforge.training.config import resolve_path

DEFAULT_STATUS = ("**Not yet trained at 2B.** The pipeline, configs and memory plan are complete, and the "
                  "measured results below come from the 3.4M-parameter CPU validation models trained from "
                  "scratch, which validate the method but are not deployable assistants. Do not publish this "
                  "card with a 2B checkpoint until it has its own evaluation results.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--status-file", default=None, help="YAML {specialist: status markdown}")
    a = p.parse_args()
    status = yaml.safe_load(open(a.status_file)) if a.status_file else {}
    os.makedirs(resolve_path("docs/model_cards"), exist_ok=True)
    for spec, info in SPEC.items():
        paths = sorted(glob.glob(resolve_path(f"results/{spec}-*/summary.json")))
        md = render(spec, status.get(spec, DEFAULT_STATUS), paths)
        out = resolve_path(f"docs/model_cards/{info['title']}.md")
        open(out, "w").write(md)
        print(f"wrote {out} ({len(paths)} result files)")

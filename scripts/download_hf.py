"""Download a TriForge export (or any model) from the Hub for offline use.

    python scripts/download_hf.py --repo <user>/TriMath-2B-ternary --out checkpoints/exports/trimath-2b
    python scripts/download_hf.py --repo microsoft/bitnet-b1.58-2B-4T-bf16 --out models/bitnet-2b4t-bf16
"""
import argparse

import _common  # noqa: F401

from triforge.huggingface.hub import download

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--revision", default=None)
    a = p.parse_args()
    print(download(a.repo, a.out, a.revision))

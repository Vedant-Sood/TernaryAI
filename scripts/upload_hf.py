"""Upload an export to a PRIVATE Hugging Face model repo (public only with --public).

    hf auth login
    python scripts/upload_hf.py --export checkpoints/exports/trimath-2b --repo <user>/TriMath-2B-ternary \
        --card docs/model_cards/TriMath.md

Refuses to upload an export whose status is not "trained" unless --allow-untrained is given, and
refuses to push into an existing public repo without --public.
"""
import argparse
import json
import os
import shutil

import _common  # noqa: F401

from triforge.huggingface.hub import upload_folder, whoami

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--export", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--card", default=None, help="model card markdown copied to README.md")
    p.add_argument("--public", action="store_true")
    p.add_argument("--allow-untrained", action="store_true")
    a = p.parse_args()
    meta = json.load(open(os.path.join(a.export, "triforge.json")))
    if meta["status"] != "trained" and not a.allow_untrained:
        raise SystemExit(f"export status is {meta['status']!r}; pass --allow-untrained to upload it anyway")
    who = whoami()
    if not who.get("ok"):
        raise SystemExit(f"not logged in to Hugging Face: {who.get('error')}")
    if a.card:
        shutil.copy(a.card, os.path.join(a.export, "README.md"))
    res = upload_folder(a.export, a.repo, f"TriForge export ({meta['specialist']}, {meta['status']})", public=a.public)
    print(f"uploaded to https://huggingface.co/{a.repo} ({'public' if a.public else 'private'}): {res}")

"""Verify Hugging Face authentication without printing the token.

    hf auth login            # once, interactive (stores the token in the HF cache, not in this repo)
    python scripts/check_hf_auth.py
"""
import json
import sys

import _common  # noqa: F401

from triforge.huggingface.hub import whoami

if __name__ == "__main__":
    info = whoami()
    print(json.dumps(info, indent=2))
    if not info.get("ok"):
        print("Not authenticated. Run `hf auth login` (or set HF_TOKEN in your shell, never in a file in this repo).")
        sys.exit(1)
    if not info.get("can_write"):
        print("Token is read-only: downloads work, uploads need a write or fine-grained token.")

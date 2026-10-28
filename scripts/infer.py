"""Local inference CLI:  python scripts/infer.py chat -m auto   (same as the `triforge` command)."""
import sys

import _common  # noqa: F401

from triforge.inference.cli import main

if __name__ == "__main__":
    main(sys.argv[1:])

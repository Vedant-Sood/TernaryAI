"""Start the local inference API:  python scripts/serve.py [--port 8765]  (same as `triforge serve`)."""
import sys

import _common  # noqa: F401

from triforge.inference.cli import main

if __name__ == "__main__":
    main(["serve", *sys.argv[1:]])

"""Shared helpers for scripts: make `src/` importable without installation and set the repo root."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("TRIFORGE_ROOT", ROOT)

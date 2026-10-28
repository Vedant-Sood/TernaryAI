"""Cleaning, de-duplication and benchmark decontamination."""

from __future__ import annotations

import ast
import hashlib
import re
import unicodedata
from collections import defaultdict

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+")


def normalize(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip().lower()


def is_malformed(ex: dict, min_chars=8, max_chars=32_000) -> str | None:
    """Return a reason string if the example should be dropped, else None."""
    p, r = ex.get("prompt", ""), ex.get("response", "")
    if not isinstance(p, str) or not isinstance(r, str):
        return "non-string field"
    if len(p.strip()) < 3 or len(r.strip()) < 1:
        return "empty prompt/response"
    total = len(p) + len(r)
    if total < min_chars or total > max_chars:
        return "length out of range"
    if "�" in p or "�" in r:
        return "unicode replacement characters"
    if sum(ch.isprintable() or ch in "\n\t" for ch in r) / max(1, len(r)) < 0.95:
        return "non-printable content"
    return None


def python_blocks_parse(text: str) -> bool:
    """True if every ```python block in `text` is syntactically valid Python."""
    blocks = re.findall(r"```(?:python|py)\n(.*?)```", text, flags=re.S)
    for b in blocks:
        try:
            ast.parse(b)
        except SyntaxError:
            return False
    return True


def exact_key(ex: dict) -> str:
    return hashlib.sha256(normalize(ex.get("prompt", "") + "\x00" + ex.get("response", "")).encode()).hexdigest()


def shingles(text: str, n: int = 5) -> set:
    words = _WORD.findall(normalize(text))
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class MinHashLSH:
    """Near-duplicate detection: MinHash signatures + banded LSH, Jaccard threshold ~0.8."""

    def __init__(self, num_perm: int = 64, bands: int = 16, seed: int = 1):
        assert num_perm % bands == 0
        self.num_perm, self.bands, self.rows = num_perm, bands, num_perm // bands
        self.salts = [hashlib.blake2b(f"{seed}-{i}".encode(), digest_size=8).digest() for i in range(num_perm)]
        self.buckets = defaultdict(list)

    def signature(self, sh: set) -> tuple:
        if not sh:
            return tuple([0] * self.num_perm)
        sig = []
        for salt in self.salts:
            sig.append(min(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8, key=salt).digest(), "little")
                           for s in sh))
        return tuple(sig)

    def query_insert(self, key, sh: set) -> bool:
        """Insert `key`; return True if it collides with an earlier item (near-duplicate)."""
        sig = self.signature(sh)
        dup = False
        for b in range(self.bands):
            band = (b, sig[b * self.rows:(b + 1) * self.rows])
            if self.buckets[band]:
                dup = True
            self.buckets[band].append(key)
        return dup


class Decontaminator:
    """Drop training examples that share any n-gram (default 13 words) with a benchmark item."""

    def __init__(self, n: int = 13):
        self.n = n
        self.grams: set = set()
        self.sources: dict = {}

    def add_benchmark(self, name: str, texts):
        before = len(self.grams)
        for t in texts:
            words = _WORD.findall(normalize(t))
            if len(words) < self.n:
                self.grams.add(" ".join(words))
                continue
            for i in range(len(words) - self.n + 1):
                self.grams.add(" ".join(words[i:i + self.n]))
        self.sources[name] = len(self.grams) - before

    def contaminated(self, text: str) -> bool:
        words = _WORD.findall(normalize(text))
        if len(words) < self.n:
            return " ".join(words) in self.grams
        return any(" ".join(words[i:i + self.n]) in self.grams for i in range(len(words) - self.n + 1))


def split_of(ex: dict, val_frac: float, test_frac: float, salt: str = "triforge") -> str:
    """Deterministic hash split, stable across runs and machines."""
    h = int(hashlib.sha256((salt + exact_key(ex)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"

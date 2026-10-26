"""Answer extraction and comparison for math / reasoning / code outputs."""

from __future__ import annotations

import re
from fractions import Fraction

_NUM = r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?"


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += len("\\boxed{")
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1].strip() if depth == 0 else None


def extract_answer(text: str) -> str | None:
    """Priority: 'Answer: X' line, GSM8K '#### X', \\boxed{X}, then the last number."""
    m = re.findall(r"Answer:\s*(.+)", text)
    if m:
        return m[-1].strip().rstrip(".")
    m = re.findall(r"####\s*(.+)", text)
    if m:
        return m[-1].strip()
    b = extract_boxed(text)
    if b is not None:
        return b
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else None


def _as_number(s: str):
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r"\\(?:d|t)?frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)", s)
    s = s.replace("(", "").replace(")", "")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace("$", "").replace(" ", "")
    return s.rstrip(".")


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = _as_number(pred), _as_number(gold)
    if a is not None and b is not None:
        return a == b
    return normalize_text(pred) == normalize_text(gold)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len)
    return text

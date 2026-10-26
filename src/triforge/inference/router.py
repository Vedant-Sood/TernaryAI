"""Rule-based router that recommends a specialist for a prompt.

It is deliberately simple, deterministic and dependency-free so it can be unit-tested and
audited. Callers can always override the recommendation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPECIALISTS = ("tricode", "trimath", "trireason")

RULES = {
    "tricode": [
        (r"```|\bdef |\bclass |#include|\bpublic static\b|=>|console\.log|\bimport \w", 3.0),
        (r"\b(python|c\+\+|java(script)?|typescript|rust|golang|sql|bash|regex|api|endpoint)\b", 2.0),
        (r"\b(code|function|bug|debug|stack ?trace|traceback|exception|compile|refactor|unit tests?|"
         r"algorithm|leetcode|codeforces|complexity|big-?o|implement|program|script)\w*", 1.5),
    ],
    "trimath": [
        (r"\\frac|\\int|\\sum|\\sqrt|\^\d|\b\d+\s*[\+\-\*/]\s*\d+", 2.0),
        (r"\b(prove|proof|lemma|theorem|integral|derivative|differentiate|integrate|matrix|eigen\w*|"
         r"polynomial|equation|solve for|gcd|lcm|prime|modulo|probability|combinatorics|geometry|"
         r"triangle|trigonometr|algebra|calculus|series|limit)\w*", 1.5),
    ],
    "trireason": [
        (r"\b(physics|velocity|acceleration|force|energy|momentum|circuit|voltage|current|resistor|"
         r"thermodynamic\w*|entropy|chemistry|reaction|molecule|biology|engineering|stress|strain|"
         r"units?|experiment|hypothesis|newton|quantum)\b", 1.5),
        (r"\b(logic|syllogism|premise|conclusion|implies|valid argument|fallacy|true or false|"
         r"explain why|reason step by step)\b", 1.5),
        (r"\b(m/s|kg|joules?|watts?|ohms?|kelvin|pascals?)\b", 1.5),
    ],
}


@dataclass
class Route:
    specialist: str
    scores: dict
    overridden: bool = False
    reason: str = ""


def score(prompt: str) -> dict:
    out = {}
    for spec, rules in RULES.items():
        s = 0.0
        for pat, w in rules:
            s += w * len(re.findall(pat, prompt, flags=re.IGNORECASE))
        out[spec] = s
    return out


def route(prompt: str, override: str | None = None, default: str = "trireason") -> Route:
    scores = score(prompt)
    if override:
        if override not in SPECIALISTS:
            raise ValueError(f"unknown specialist {override!r}; choose from {SPECIALISTS}")
        return Route(override, scores, overridden=True, reason="user override")
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return Route(default, scores, reason="no domain cues; default specialist")
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - ranked[1]
    return Route(best, scores, reason=f"highest cue score (margin {margin:.1f})")

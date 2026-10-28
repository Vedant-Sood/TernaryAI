"""Procedurally generated, automatically verified task data.

Why this exists: it gives every specialist a training and evaluation signal with *exactly
known answers*, no licensing questions and no benchmark contamination. It is what the
small CPU validation runs train on, and it doubles as an independent held-out test set
next to public benchmarks. It is not a substitute for real instruction data at scale.

Every generator is a pure function of a seeded random.Random, so data is reproducible.
Each example: {"prompt": str, "response": str, "answer": str, "task": str}.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

# ----------------------------------------------------------------------------- TriMath


def _arith_chain(r: random.Random):
    n = r.randint(2, 4)
    vals = [r.randint(2, 99) for _ in range(n + 1)]
    ops = [r.choice("+-*") for _ in range(n)]
    expr = str(vals[0])
    steps, acc = [], vals[0]
    for op, v in zip(ops, vals[1:]):
        new = acc + v if op == "+" else acc - v if op == "-" else acc * v
        steps.append(f"{acc} {op} {v} = {new}")
        expr = f"({expr} {op} {v})"
        acc = new
    return {"prompt": f"Compute {expr}.", "response": "\n".join(steps) + f"\nAnswer: {acc}",
            "answer": str(acc), "task": "math/arithmetic"}


def _linear_eq(r: random.Random):
    a = r.choice([i for i in range(-9, 10) if i not in (0,)])
    x = r.randint(-20, 20)
    b = r.randint(-50, 50)
    c = a * x + b
    resp = (f"Subtract {b} from both sides: {a}x = {c} - ({b}) = {c - b}\n"
            f"Divide by {a}: x = {c - b}/{a} = {x}\nAnswer: {x}")
    return {"prompt": f"Solve for x: {a}x + ({b}) = {c}", "response": resp, "answer": str(x),
            "task": "math/linear_equation"}


def _gcd_lcm(r: random.Random):
    g = r.randint(2, 15)
    a, b = g * r.randint(2, 20), g * r.randint(2, 20)
    gg = math.gcd(a, b)
    if r.random() < 0.5:
        steps, x, y = [], a, b
        while y:
            steps.append(f"{x} = {x // y}*{y} + {x % y}")
            x, y = y, x % y
        return {"prompt": f"What is gcd({a}, {b})?", "response": "\n".join(steps) + f"\nAnswer: {gg}",
                "answer": str(gg), "task": "math/gcd"}
    lcm = a * b // gg
    return {"prompt": f"What is lcm({a}, {b})?",
            "response": f"gcd({a}, {b}) = {gg}\nlcm = {a}*{b}/{gg} = {lcm}\nAnswer: {lcm}",
            "answer": str(lcm), "task": "math/lcm"}


def _fractions(r: random.Random):
    p = Fraction(r.randint(1, 12), r.randint(2, 12))
    q = Fraction(r.randint(1, 12), r.randint(2, 12))
    op = r.choice("+-*")
    res = p + q if op == "+" else p - q if op == "-" else p * q
    return {"prompt": f"Simplify {p} {op} {q}.", "response": f"{p} {op} {q} = {res}\nAnswer: {res}",
            "answer": str(res), "task": "math/fractions"}


def _derivative(r: random.Random):
    coeffs = [r.randint(-6, 6) for _ in range(r.randint(2, 4))]
    if all(c == 0 for c in coeffs[1:]):
        coeffs[1] = 1

    def poly(cs):
        terms = []
        for p, c in reversed(list(enumerate(cs))):
            if c == 0:
                continue
            terms.append(f"{c}" if p == 0 else f"{c}x" if p == 1 else f"{c}x^{p}")
        return " + ".join(terms).replace("+ -", "- ") or "0"

    d = [p * c for p, c in enumerate(coeffs)][1:]
    ans = poly(d)
    return {"prompt": f"Differentiate f(x) = {poly(coeffs)}.",
            "response": f"Apply the power rule term by term.\nf'(x) = {ans}\nAnswer: {ans}",
            "answer": ans, "task": "math/derivative"}


def _word_problem(r: random.Random):
    name = r.choice(["Asha", "Ben", "Chen", "Dana", "Eli", "Farah"])
    item = r.choice(["apples", "books", "pencils", "stamps"])
    a, b, c = r.randint(5, 60), r.randint(2, 9), r.randint(1, 20)
    total = a * b - c
    return {"prompt": f"{name} buys {b} boxes with {a} {item} each and gives away {c}. "
                      f"How many {item} are left?",
            "response": f"{b} boxes * {a} = {a * b} {item}\n{a * b} - {c} = {total}\nAnswer: {total}",
            "answer": str(total), "task": "math/word_problem"}


MATH_TASKS = [_arith_chain, _linear_eq, _gcd_lcm, _fractions, _derivative, _word_problem]

# ----------------------------------------------------------------------------- TriReason


def _kinematics(r: random.Random):
    v0, a, t = r.randint(0, 40), r.randint(1, 12), r.randint(1, 20)
    s = v0 * t + Fraction(1, 2) * a * t * t
    s = int(s) if s.denominator == 1 else float(s)
    return {"prompt": f"A cart starts at {v0} m/s and accelerates uniformly at {a} m/s^2 for {t} s. "
                      "How far does it travel, in metres?",
            "response": f"s = v0*t + a*t^2/2 = {v0}*{t} + {a}*{t}^2/2 = {v0 * t} + {a * t * t / 2:g} = {s:g}\n"
                        f"Answer: {s:g}",
            "answer": f"{s:g}", "task": "reason/kinematics"}


def _ohms_law(r: random.Random):
    R, I = r.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 100]), r.randint(1, 40)
    V = R * I
    kind = r.randrange(3)
    if kind == 0:
        q, resp, ans = f"A {R} ohm resistor carries {I} A. What is the voltage across it, in volts?", \
            f"V = I*R = {I}*{R} = {V}", V
    elif kind == 1:
        q, resp, ans = f"A {V} V source drives a {R} ohm resistor. What current flows, in amperes?", \
            f"I = V/R = {V}/{R} = {I}", I
    else:
        q, resp, ans = f"A resistor carries {I} A at {V} V. What is its power dissipation, in watts?", \
            f"P = V*I = {V}*{I} = {V * I}", V * I
    return {"prompt": q, "response": f"{resp}\nAnswer: {ans}", "answer": str(ans), "task": "reason/circuits"}


_UNITS = [("km", "m", 1000), ("m", "cm", 100), ("kg", "g", 1000), ("h", "min", 60), ("min", "s", 60),
          ("L", "mL", 1000)]


def _units(r: random.Random):
    a, b, f = r.choice(_UNITS)
    x = r.randint(1, 999)
    return {"prompt": f"Convert {x} {a} to {b}.", "response": f"1 {a} = {f} {b}, so {x} {a} = {x}*{f} = {x * f} {b}\n"
                                                             f"Answer: {x * f}",
            "answer": str(x * f), "task": "reason/units"}


def _logic(r: random.Random):
    vars_ = ["p", "q", "r"]
    vals = {v: r.choice([True, False]) for v in vars_}
    a, b, c = r.sample(vars_, 3)
    form = r.randrange(3)
    if form == 0:
        expr, val = f"({a} and {b}) or not {c}", (vals[a] and vals[b]) or not vals[c]
    elif form == 1:
        expr, val = f"not ({a} or {b}) or {c}", (not (vals[a] or vals[b])) or vals[c]
    else:
        expr, val = f"({a} implies {b}) and {c}", ((not vals[a]) or vals[b]) and vals[c]
    given = ", ".join(f"{v} = {vals[v]}" for v in vars_)
    return {"prompt": f"Given {given}, is \"{expr}\" true or false?",
            "response": f"Substitute the values and evaluate step by step.\nAnswer: {str(val).lower()}",
            "answer": str(val).lower(), "task": "reason/logic"}


def _syllogism(r: random.Random):
    A, B, C = r.sample(["mammals", "metals", "conductors", "animals", "birds", "planets", "solids",
                        "stars", "reptiles", "liquids"], 3)
    valid = r.random() < 0.5
    if valid:
        q = f"All {A} are {B}. All {B} are {C}. Does it follow that all {A} are {C}?"
        resp, ans = "Transitivity of 'all': A ⊆ B and B ⊆ C imply A ⊆ C.", "yes"
    else:
        q = f"All {A} are {B}. Some {B} are {C}. Does it follow that all {A} are {C}?"
        resp, ans = "'Some B are C' does not cover all of B, so A may lie outside C.", "no"
    return {"prompt": q, "response": f"{resp}\nAnswer: {ans}", "answer": ans, "task": "reason/syllogism"}


def _energy(r: random.Random):
    m, h = r.randint(1, 100), r.randint(1, 200)
    E = m * 10 * h
    return {"prompt": f"A {m} kg mass is lifted {h} m. Using g = 10 m/s^2, how much potential energy does it "
                      "gain, in joules?",
            "response": f"E = m*g*h = {m}*10*{h} = {E}\nAnswer: {E}", "answer": str(E), "task": "reason/energy"}


def _momentum(r: random.Random):
    m, v = r.randint(1, 200), r.randint(1, 60)
    return {"prompt": f"What is the momentum of a {m} kg object moving at {v} m/s, in kg*m/s?",
            "response": f"p = m*v = {m}*{v} = {m * v}\nAnswer: {m * v}", "answer": str(m * v), "task": "reason/momentum"}


def _wave(r: random.Random):
    f, lam = r.randint(1, 500), r.randint(1, 40)
    if r.random() < 0.5:
        return {"prompt": f"A wave has frequency {f} Hz and wavelength {lam} m. What is its speed, in m/s?",
                "response": f"v = f*lambda = {f}*{lam} = {f * lam}\nAnswer: {f * lam}", "answer": str(f * lam),
                "task": "reason/waves"}
    v = f * lam
    return {"prompt": f"A wave travels at {v} m/s with wavelength {lam} m. What is its frequency, in Hz?",
            "response": f"f = v/lambda = {v}/{lam} = {f}\nAnswer: {f}", "answer": str(f), "task": "reason/waves"}


def _work_power(r: random.Random):
    F, d, t = r.randint(1, 300), r.randint(1, 100), r.choice([1, 2, 4, 5, 10, 20])
    W = F * d
    P = Fraction(W, t)
    P = int(P) if P.denominator == 1 else float(P)
    return {"prompt": f"A constant {F} N force pushes a box {d} m in {t} s. What average power is delivered, in watts?",
            "response": f"W = F*d = {F}*{d} = {W} J\nP = W/t = {W}/{t} = {P:g}\nAnswer: {P:g}", "answer": f"{P:g}",
            "task": "reason/work_power"}


REASON_TASKS = [_kinematics, _ohms_law, _units, _logic, _syllogism, _energy, _momentum, _wave, _work_power]

# ----------------------------------------------------------------------------- TriCode

_FUNCS = [
    ("sum of squares", "def f(xs):\n    return sum(x * x for x in xs)"),
    ("count evens", "def f(xs):\n    return sum(1 for x in xs if x % 2 == 0)"),
    ("max minus min", "def f(xs):\n    return max(xs) - min(xs)"),
    ("reverse and take first two", "def f(xs):\n    return xs[::-1][:2]"),
    ("running total", "def f(xs):\n    out, t = [], 0\n    for x in xs:\n        t += x\n        out.append(t)\n    return out"),
    ("dedupe keep order", "def f(xs):\n    seen, out = set(), []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out"),
    ("sorted descending", "def f(xs):\n    return sorted(xs, reverse=True)"),
    ("index of max", "def f(xs):\n    return xs.index(max(xs))"),
]


def _code_output(r: random.Random):
    _, src = r.choice(_FUNCS)
    xs = [r.randint(-9, 9) for _ in range(r.randint(3, 6))]
    ns: dict = {}
    exec(src, ns)  # trusted, fixed source above
    out = ns["f"](list(xs))
    return {"prompt": f"What does this Python code print?\n```python\n{src}\nprint(f({xs}))\n```",
            "response": f"Trace f on {xs}.\nAnswer: {out}", "answer": str(out), "task": "code/output"}


_IMPL = [
    ("Write a Python function `add(a, b)` that returns the sum of a and b.",
     "def add(a, b):\n    return a + b", ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]),
    ("Write a Python function `is_even(n)` that returns True if n is even.",
     "def is_even(n):\n    return n % 2 == 0", ["assert is_even(4)", "assert not is_even(7)"]),
    ("Write a Python function `square_all(xs)` that returns a list of the squares of xs.",
     "def square_all(xs):\n    return [x * x for x in xs]", ["assert square_all([1, 2, 3]) == [1, 4, 9]"]),
    ("Write a Python function `count_char(s, c)` that counts occurrences of character c in s.",
     "def count_char(s, c):\n    return s.count(c)", ["assert count_char('banana', 'a') == 3"]),
    ("Write a Python function `factorial(n)` that returns n! for n >= 0.",
     "def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out",
     ["assert factorial(0) == 1", "assert factorial(5) == 120"]),
    ("Write a Python function `is_palindrome(s)` that checks whether s reads the same backwards.",
     "def is_palindrome(s):\n    return s == s[::-1]", ["assert is_palindrome('abba')", "assert not is_palindrome('ab')"]),
    ("Write a Python function `fib(n)` that returns the n-th Fibonacci number with fib(0) = 0.",
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     ["assert fib(0) == 0", "assert fib(10) == 55"]),
    ("Write a Python function `clamp(x, lo, hi)` that limits x to the range [lo, hi].",
     "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))", ["assert clamp(5, 0, 3) == 3", "assert clamp(-1, 0, 3) == 0"]),
]


def _code_impl(r: random.Random):
    prompt, sol, tests = r.choice(_IMPL)
    return {"prompt": prompt, "response": f"```python\n{sol}\n```", "answer": sol, "tests": tests,
            "task": "code/implement"}


CODE_TASKS = [_code_output, _code_output, _code_impl]

GENERATORS = {"trimath": MATH_TASKS, "trireason": REASON_TASKS, "tricode": CODE_TASKS}


def generate(specialist: str, n: int, seed: int) -> list[dict]:
    r = random.Random(seed)
    tasks = GENERATORS[specialist]
    return [tasks[i % len(tasks)](r) for i in range(n)]


def generate_split(specialist: str, n_train: int, n_test: int, seed: int = 0):
    """Train and test from disjoint seeds, with exact-prompt overlap removed from train."""
    test = generate(specialist, n_test, seed=10_000 + seed)
    test_prompts = {e["prompt"] for e in test}
    train = [e for e in generate(specialist, n_train, seed=seed) if e["prompt"] not in test_prompts]
    return train, test


def run_python_tests(code: str, tests: list[str], timeout: float = 5.0) -> bool:
    """Execute candidate code + asserts in a subprocess with a timeout (see evaluation.code_exec)."""
    from ..evaluation.code_exec import run_code
    return run_code(code + "\n" + "\n".join(tests), timeout=timeout)["passed"]

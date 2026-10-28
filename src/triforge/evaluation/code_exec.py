"""Execute generated code against tests in a separate, resource-limited process.

This is a best-effort sandbox (isolated interpreter, temp dir, timeout, memory and CPU-time
limits on Linux/WSL). It is not a security boundary: only run it on model outputs inside a
VM, container or WSL instance you are willing to lose.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile


def _limits(mem_mb):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2 ** 20, mem_mb * 2 ** 20))
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            os.setsid()
        except Exception:
            pass
    return apply


def run_code(code: str, timeout: float = 10.0, mem_mb: int = 1024) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits(mem_mb) if os.name == "posix" else None,
                               env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
            return {"passed": p.returncode == 0, "returncode": p.returncode,
                    "stdout": p.stdout[-2000:], "stderr": p.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {"passed": False, "returncode": None, "stdout": "", "stderr": "timeout"}

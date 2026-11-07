"""Execute generated code against tests in a separate, resource-limited process.

This is a best-effort sandbox (isolated interpreter, temp dir, timeout, memory and CPU-time
limits on Linux/WSL). It is not a security boundary: only run it on model outputs inside a
VM, container or WSL instance you are willing to lose.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile


def _limits(mem_mb):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2 ** 20, mem_mb * 2 ** 20))
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 2 ** 20, 64 * 2 ** 20))
        except Exception:
            pass
    return apply


def _kill_group(p):
    """Kill the program and anything it spawned (it runs in its own session / process group)."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, AttributeError):
        p.kill()


def run_code(code: str, timeout: float = 10.0, mem_mb: int = 1024) -> dict:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        with open(path, "w") as f:
            f.write(code)
        posix = os.name == "posix"
        p = subprocess.Popen([sys.executable, "-I", path], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=posix, preexec_fn=_limits(mem_mb) if posix else None,
                             env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
        try:
            out, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(p)
            try:
                p.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # a detached grandchild still holds the pipes
                pass
            return {"passed": False, "returncode": None, "stdout": "", "stderr": "timeout"}
        finally:
            if posix:  # reap background children the program left behind
                _kill_group(p)
        return {"passed": p.returncode == 0, "returncode": p.returncode,
                "stdout": out[-2000:], "stderr": err[-2000:]}

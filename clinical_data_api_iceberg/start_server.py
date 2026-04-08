#!/usr/bin/env python
"""
Cross-platform server launcher for the Clinical Data API.

On Linux / macOS / WSL
-----------------------
Delegates to Gunicorn with UvicornWorker and the settings in
``gunicorn.conf.py``.  Gunicorn's master process supervises all workers,
handles graceful restarts, and (with ``preload_app=True``) allows the OS to
share read-only Arrow buffers across workers via copy-on-write.

On Windows
----------
Gunicorn is not available on Windows (it uses POSIX-only syscalls such as
``fcntl`` and ``os.fork``).  This launcher falls back to Uvicorn's built-in
multi-process mode (``--workers N``), which uses Python's ``multiprocessing``
module with the ``spawn`` start method.

.. warning::
   On Windows each worker process is spawned independently and loads its own
   copy of the dataset from disk.  Memory is **not** shared between workers.
   Plan for approximately ``N_workers × dataset_size`` RAM.  To keep the
   default safe, the Windows worker count is capped at ``min(4, CPU_COUNT)``
   instead of the usual ``(2 × CPU_COUNT) + 1`` used on Linux/macOS.

Usage
-----
::

    # Auto-detect OS and launch with defaults
    python start_server.py

    # Override worker count
    CLINICAL_WORKERS=4 python start_server.py       # Linux / macOS
    set CLINICAL_WORKERS=4 && python start_server.py  # Windows CMD
    $env:CLINICAL_WORKERS=4; python start_server.py  # PowerShell

Environment variables
---------------------
All variables are shared with ``gunicorn.conf.py`` so the same ``.env`` file
works on every platform.

+--------------------+--------------------+-----------------------------------+
| Variable           | Default            | Description                       |
+====================+====================+===================================+
| CLINICAL_BIND      | 0.0.0.0:8090       | ``host:port`` to listen on        |
+--------------------+--------------------+-----------------------------------+
| CLINICAL_WORKERS   | (2×CPU)+1 *        | Number of worker processes        |
+--------------------+--------------------+-----------------------------------+
| * Windows default  | min(4, CPU)        | Avoids N × dataset RAM on spawn   |
+--------------------+--------------------+-----------------------------------+
| CLINICAL_TIMEOUT   | 120                | Worker silence timeout (seconds)  |
+--------------------+--------------------+-----------------------------------+
| CLINICAL_LOG_LEVEL | info               | Logging verbosity                 |
+--------------------+--------------------+-----------------------------------+
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent  # clinical_data_api/


def _env_int(name: str, default: int) -> int:
    """Return the integer value of an environment variable, or *default*."""
    raw = os.environ.get(name, "").strip()
    if raw.isdigit():
        return int(raw)
    return default


def _resolve_bind() -> tuple[str, int]:
    """Parse ``CLINICAL_BIND`` into *(host, port)*."""
    bind = os.environ.get("CLINICAL_BIND", "0.0.0.0:8090").strip()
    if ":" in bind:
        host, _, port_str = bind.rpartition(":")
        try:
            return host, int(port_str)
        except ValueError:
            pass
    return bind, 8090


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_IS_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Configuration (mirrors gunicorn.conf.py)
# ---------------------------------------------------------------------------

_cpu = multiprocessing.cpu_count()
# On Windows each worker is independently *spawned* (no fork/COW), so every
# worker loads the full dataset from disk into its own private address space.
# Using the standard (2×CPU)+1 heuristic on a 16-core machine would spin up
# 33 workers × ~150 MB ≈ ~5 GB of RAM just for the cached data alone.
# Cap the Windows default at min(4, cpu_count) so the service starts safely
# on ordinary development machines.  Production operators can always raise it
# with: set CLINICAL_WORKERS=8 (CMD) / $env:CLINICAL_WORKERS=8 (PowerShell).
_default_workers: int = min(4, _cpu) if _IS_WINDOWS else (_cpu * 2 + 1)
WORKERS: int = _env_int("CLINICAL_WORKERS", _default_workers)
TIMEOUT: int = _env_int("CLINICAL_TIMEOUT", 120)
LOG_LEVEL: str = os.environ.get("CLINICAL_LOG_LEVEL", "info").strip().lower()
HOST, PORT = _resolve_bind()
APP: str = "app.main:app"


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def _run_gunicorn() -> None:
    """Launch Gunicorn with UvicornWorker (Linux / macOS / WSL)."""
    conf = _HERE / "gunicorn.conf.py"
    cmd: list[str] = [
        sys.executable,
        "-m",
        "gunicorn",
        APP,
        "--config",
        str(conf),
    ]

    # Allow the caller to override individual settings via CLI env vars
    # (gunicorn.conf.py already reads the same env vars, so these are only
    # needed if someone bypasses the conf file).
    print(
        f"[start_server] Launching Gunicorn  "
        f"workers={WORKERS}  bind={HOST}:{PORT}  "
        f"loglevel={LOG_LEVEL}  timeout={TIMEOUT}s",
        flush=True,
    )
    _exec(cmd)


def _run_uvicorn() -> None:
    """Launch Uvicorn with multiple workers (Windows / cross-platform fallback)."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "uvicorn",
        APP,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--workers",
        str(WORKERS),
        "--log-level",
        LOG_LEVEL,
        "--timeout-keep-alive",
        "5",
    ]

    print(
        f"[start_server] Launching Uvicorn   "
        f"workers={WORKERS}  bind={HOST}:{PORT}  "
        f"loglevel={LOG_LEVEL}",
        flush=True,
    )
    print(
        "[start_server] NOTE: On Windows each worker loads the full dataset "
        "independently — memory is NOT shared between workers.",
        flush=True,
    )
    _exec(cmd)


def _exec(cmd: list[str]) -> None:
    """
    Replace the current process with *cmd* (Unix) or run it as a subprocess
    (Windows, which does not support ``os.execvp``).
    """
    if _IS_WINDOWS:
        # ``os.execvp`` is not available on Windows; use subprocess so that
        # Ctrl-C / SIGINT is forwarded correctly.
        try:
            proc = subprocess.run(cmd, cwd=str(_HERE))
            sys.exit(proc.returncode)
        except KeyboardInterrupt:
            sys.exit(0)
    else:
        # On Unix, replace this process image entirely — no Python wrapper
        # sitting idle in memory.
        os.execvp(cmd[0], cmd)  # noqa: S606


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print(
        f"[start_server] Platform={platform.system()}  "
        f"Python={sys.version.split()[0]}  "
        f"CPUs={_cpu}",
        flush=True,
    )

    # Sanity-check: make sure we are running from the right directory so that
    # ``app.main`` resolves correctly.
    if not (_HERE / "app" / "main.py").exists():
        print(
            f"[start_server] ERROR: cannot find app/main.py under '{_HERE}'.\n"
            "Run this script from the 'clinical_data_api/' directory:\n"
            "    python start_server.py",
            file=sys.stderr,
        )
        sys.exit(1)

    # Change to the project root so relative imports inside the app work
    # regardless of where the caller invoked this script from.
    os.chdir(_HERE)

    if _IS_WINDOWS:
        _run_uvicorn()
    else:
        # Try Gunicorn first; fall back gracefully if it is not installed.
        try:
            import gunicorn  # noqa: F401

            _run_gunicorn()
        except ModuleNotFoundError:
            print(
                "[start_server] WARNING: gunicorn not found — falling back to uvicorn.",
                flush=True,
            )
            _run_uvicorn()


if __name__ == "__main__":
    main()

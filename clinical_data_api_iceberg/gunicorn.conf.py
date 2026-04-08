"""
Gunicorn configuration for the Clinical Data API.

Usage (Linux / macOS / WSL)
---------------------------
From the ``clinical_data_api/`` directory:

    gunicorn app.main:app -c gunicorn.conf.py

Or with env overrides:

    CLINICAL_WORKERS=8 gunicorn app.main:app -c gunicorn.conf.py

Windows note
------------
Gunicorn requires a Unix-based OS (it uses ``os.fork()``).  On Windows run
the API with ``uvicorn`` directly (it supports ``--workers`` since v0.20):

    uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8090

Design decisions
----------------
* **UvicornWorker** – wraps uvicorn inside a Gunicorn worker process so we
  get Gunicorn's process supervision AND uvicorn's fast ASGI event loop.

* **Worker count** – the standard heuristic is ``(2 × CPU_COUNT) + 1``.
  For CPU-bound work (e.g. serialising 3 000 records) this saturates all
  cores; the +1 ensures at least one worker is always ready to accept new
  connections while others are serialising.  Override with ``CLINICAL_WORKERS``.

* **preload_app = True** – Gunicorn imports the Python module *once* in the
  master process before forking workers.  On Linux/macOS the OS copy-on-write
  mechanism means workers share the master's virtual memory pages until they
  write to them.  Since our in-memory DataFrames are read-only after loading,
  the Arrow column buffers (the bulk of memory) remain physically shared.
  This makes memory usage roughly ``1 × dataset_size`` rather than
  ``N_workers × dataset_size``.

  ⚠ The ASGI lifespan (data loading) still runs in *each* worker because
  Gunicorn spawns workers after import but the lifespan is ASGI-level.
  If you want single-load behaviour, move ``cache.load_all_domains()`` to
  module scope in ``app/main.py`` (before the ``FastAPI()`` call) and
  disable it inside the lifespan.

* **timeout = 120** – the first request after a cold-cache restart may take
  several seconds while data loads.  120 s gives ample headroom.

* **worker_tmp_dir = "/dev/shm"** – on Linux, using the in-memory tmpfs for
  Gunicorn's worker heartbeat file eliminates the disk I/O for liveness probes.
"""

from __future__ import annotations

import multiprocessing
import os

# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

bind: str = os.getenv("CLINICAL_BIND", "0.0.0.0:8090")
backlog: int = 2048

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

_cpu = multiprocessing.cpu_count()
workers: int = int(os.getenv("CLINICAL_WORKERS", str(_cpu * 2 + 1)))
worker_class: str = "uvicorn.workers.UvicornWorker"
worker_connections: int = 1000

# On Linux use in-memory tmpfs for heartbeat files (faster, no disk I/O).
# Falls back gracefully on macOS / WSL where /dev/shm may not exist.
worker_tmp_dir: str | None = "/dev/shm" if os.path.isdir("/dev/shm") else None

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------

timeout: int = int(os.getenv("CLINICAL_TIMEOUT", "120"))
graceful_timeout: int = 30
keepalive: int = 5

# ---------------------------------------------------------------------------
# App loading
# ---------------------------------------------------------------------------

# Load app once in the master process before forking.
# On Linux/macOS workers inherit pages via OS copy-on-write.
preload_app: bool = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

loglevel: str = os.getenv("CLINICAL_LOG_LEVEL", "info")
accesslog: str = "-"  # stdout
errorlog: str = "-"  # stdout
access_log_format: str = '%(h)s "%(r)s" %(s)s %(b)sB %(D)sμs'

# ---------------------------------------------------------------------------
# Process naming
# ---------------------------------------------------------------------------

proc_name: str = "clinical-data-api"

# ---------------------------------------------------------------------------
# Server hooks
# ---------------------------------------------------------------------------


def on_starting(server) -> None:  # noqa: ANN001
    server.log.info(
        "Clinical Data API – Gunicorn starting  workers=%d  bind=%s",
        workers,
        bind,
    )


def post_fork(server, worker) -> None:  # noqa: ANN001
    server.log.info("Worker spawned  pid=%d", worker.pid)


def worker_exit(server, worker) -> None:  # noqa: ANN001
    server.log.info("Worker exited  pid=%d", worker.pid)

"""
Gunicorn configuration for production deployment.

Usage:
    gunicorn -c gunicorn.conf.py app.main:app

Gunicorn spawns multiple worker processes (each running Uvicorn's async event
loop), giving true OS-level multiprocessing for handling concurrent requests.

DuckDB is opened in read-only mode per-thread, so multiple workers can safely
read the same database file simultaneously.
"""

# ---------------------------------------------------------------------------
# Server socket
# ---------------------------------------------------------------------------
bind = "0.0.0.0:8090"

# ---------------------------------------------------------------------------
# Worker settings
# ---------------------------------------------------------------------------
# UvicornWorker runs a full asyncio event loop per worker process.
# Each worker is a separate OS process — proper multiprocessing.
worker_class = "uvicorn.workers.UvicornWorker"

# Single worker to avoid DuckDB file-lock contention across processes
workers = 4

# Threads per worker (for blocking I/O bursts within a worker)
threads = 8

# Worker connections (max simultaneous connections per worker)
worker_connections = 1000

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------
timeout = 60  # Kill worker if it doesn't respond in N seconds
keepalive = 5  # How long to wait between keepalive connections
graceful_timeout = 30  # Time allowed for in-flight requests on SIGTERM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
accesslog = "-"  # stdout
errorlog = "-"  # stderr
loglevel = "info"
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sμs'
)

# ---------------------------------------------------------------------------
# Process naming
# ---------------------------------------------------------------------------
proc_name = "clinical-data-api"

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
preload_app = True  # Load app code before forking workers (saves memory via CoW)
max_requests = 1000  # Recycle worker after N requests (prevents memory leaks)
max_requests_jitter = 100  # Random jitter to avoid thundering-herd on recycling

"""
DuckDB connection management.

Design
------
DuckDB's file-level locking means that opening *multiple* connections to the
same database file (whether from different threads or different processes) can
cause lock contention and "database is locked" errors.

To avoid this we keep exactly ONE connection per process, protected by a
threading.Lock.  Any thread that needs the DB acquires the lock first; all
other threads simply wait in line.  Because callers run inside
asyncio.to_thread(), their waiting happens off the event loop — the async
event loop stays free to serve other requests while a thread holds the lock.

Result
------
- 1 file handle per process → no cross-process or cross-thread locking.
- threading.Lock → safe concurrent access from asyncio.to_thread workers.
- Event loop is never blocked (callers must use asyncio.to_thread).
"""

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import duckdb

from app.config import get_settings

logger = logging.getLogger(__name__)

# One lock and one connection per process.
# After a fork (Gunicorn workers) each child gets its own independent copy of
# both variables because OS-level fork duplicates the address space but does
# NOT share state with the parent — so there is truly one connection per worker.
_db_lock: threading.Lock = threading.Lock()
_process_conn: Optional[duckdb.DuckDBPyConnection] = None


def _get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return the process-level DuckDB connection, creating it on first call.

    MUST be called with _db_lock already held.
    """
    global _process_conn

    if _process_conn is None:
        settings = get_settings()
        db_path = settings.db_path

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"DuckDB database not found at: {db_path}. "
                "Ensure 'clinical_data.duckdb' is present in the project root."
            )

        logger.info(
            "Opening DuckDB connection for process %d (thread %s)",
            os.getpid(),
            threading.current_thread().name,
        )
        _process_conn = duckdb.connect(database=db_path, read_only=True)

    return _process_conn


@contextmanager
def get_db() -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Yield the process-level DuckDB connection under the process lock.

    Usage (inside a worker thread via asyncio.to_thread):
        with get_db() as conn:
            result = conn.execute("SELECT ...").fetchall()

    The lock is held for the duration of the with-block, serialising all DB
    access within this process.  This is intentional: it is far preferable to
    queue queries than to race for the file lock and get errors.
    """
    with _db_lock:
        conn = _get_connection()
        try:
            yield conn
        except duckdb.Error as exc:
            logger.error("DuckDB query error: %s", exc)
            # Discard the broken connection so the next caller gets a fresh one.
            _reset_connection()
            raise


def _reset_connection() -> None:
    """Close and discard the process-level connection (lock must be held)."""
    global _process_conn
    if _process_conn is not None:
        try:
            _process_conn.close()
        except Exception:
            pass
        _process_conn = None
        logger.warning(
            "DuckDB connection reset for process %d after error.",
            os.getpid(),
        )


def close_connection() -> None:
    """
    Explicitly close the process-level connection.

    Useful in application teardown, tests, or post-fork hooks.
    Safe to call even if no connection is open.
    """
    with _db_lock:
        if _process_conn is not None:
            _reset_connection()
            logger.info(
                "DuckDB connection closed for process %d.",
                os.getpid(),
            )

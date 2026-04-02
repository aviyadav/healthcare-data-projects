"""
DuckDB connection management.

DuckDB supports multiple concurrent read-only connections across processes/threads.
We use thread-local storage so each worker thread gets its own connection,
avoiding any cross-thread state sharing while keeping connections open and warm.
"""
import threading
import logging
from contextlib import contextmanager
from pathlib import Path

import duckdb

from app.config import get_settings

logger = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_connection() -> duckdb.DuckDBPyConnection:
    """
    Return the thread-local DuckDB connection, creating it if needed.
    Opens the database in read-only mode so multiple processes can share it safely.
    """
    if not hasattr(_thread_local, "conn") or _thread_local.conn is None:
        settings = get_settings()
        db_path = settings.db_path

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"DuckDB database not found at: {db_path}. "
                "Ensure 'clinical_data.duckdb' is present in the project root."
            )

        logger.info("Opening DuckDB connection for thread %s", threading.current_thread().name)
        _thread_local.conn = duckdb.connect(database=db_path, read_only=True)

    return _thread_local.conn


@contextmanager
def get_db():
    """
    Yield a DuckDB connection.
    The connection is thread-local and long-lived (not closed after each request),
    which avoids repeated open/close overhead while remaining safe under Gunicorn
    workers since each worker process has isolated memory.
    """
    conn = _get_connection()
    try:
        yield conn
    except duckdb.Error as exc:
        logger.error("DuckDB query error: %s", exc)
        # Attempt to recover by discarding the broken connection
        _thread_local.conn = None
        raise


def close_connection() -> None:
    """Close and discard the thread-local connection (useful in teardown/tests)."""
    if hasattr(_thread_local, "conn") and _thread_local.conn is not None:
        try:
            _thread_local.conn.close()
        except Exception:
            pass
        _thread_local.conn = None
        logger.info("Closed DuckDB connection for thread %s", threading.current_thread().name)

"""
In-memory domain cache.

Problem
-------
On every API request the original reader called ``pyarrow.dataset.dataset()``
to discover Parquet files (≈3 s for large domains) and then ``to_table()``
to read them from disk (≈3 s more).  That made every request take 5-10 s,
regardless of page size.

Solution
--------
At application startup each of the six SDTM domains is loaded **once** from
the Hive-partitioned Parquet tree into a Polars DataFrame that lives in
process memory for the lifetime of the server.  Subsequent requests filter
and slice the in-memory DataFrame, reducing per-request latency to ~5-30 ms
even for ``page_size=3000``.

Loading strategy
----------------
* All six domains are loaded **in parallel** via a ``ThreadPoolExecutor``
  (I/O-bound, benefits from concurrency even with the GIL).
* ``pyarrow.dataset`` with ``partitioning="hive"`` is used for file discovery
  and reading – it resolves partition columns (STUDYID, SITEID, USUBJID, …)
  from directory path segments without duplicating them from file data.
* ``pl.from_arrow()`` converts the Arrow table to a Polars DataFrame in a
  zero-copy fashion where possible.
* Dictionary-encoded (``Categorical``) columns are cast to plain ``Utf8``
  once at load time so that filter comparisons and ``to_dicts()`` are cheap
  on every request.

Gunicorn + preload_app
----------------------
When Gunicorn is started with ``preload_app = True`` the master process
imports this module and forks workers **after** the module-level code runs.
Because the data is loaded inside the ASGI lifespan (``app/main.py``) and
*not* at import time, each worker starts its own lifespan and loads the data
independently.  This is safe: the data is read-only and the Arrow column
buffers (the bulk of RAM) are eligible for OS-level copy-on-write sharing on
Linux/macOS after fork.  On Windows (spawn-based multiprocessing) each worker
loads independently – memory is not shared but correctness is unaffected.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level registry
# ---------------------------------------------------------------------------

# Populated by load_all_domains(); read-only afterwards.
_DOMAIN_FRAMES: dict[str, pl.DataFrame] = {}

# Per-domain load stats (timing + row counts) surfaced by /health.
_LOAD_STATS: dict[str, dict[str, Any]] = {}

# All six SDTM domain codes handled by this API.
DOMAINS: tuple[str, ...] = ("AE", "CM", "DM", "LB", "TV", "VS")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_single_domain(domain: str, data_root: Path) -> None:
    """
    Load one SDTM domain from Hive-partitioned Parquet into a Polars DataFrame.

    Steps
    -----
    1. Open a PyArrow dataset (resolves partition columns from directory names).
    2. Read all files into an Arrow Table (``to_table()`` – bulk I/O).
    3. Convert to a Polars DataFrame (``pl.from_arrow()``).
    4. Cast every ``Categorical`` / ``Enum`` column to plain ``Utf8`` so that
       equality predicates and ``to_dicts()`` are fast on every subsequent
       request.
    5. Store the result in the module-level ``_DOMAIN_FRAMES`` dict.

    Parameters
    ----------
    domain:
        SDTM domain code, e.g. ``"LB"``.
    data_root:
        Root directory that contains one sub-folder per domain.
    """
    domain_path = data_root / domain
    if not domain_path.exists():
        logger.warning("Cache: domain path not found – %s (skipped)", domain_path)
        return

    t0 = time.perf_counter()
    logger.info("Cache: loading domain %s …", domain)

    # ── 1 & 2. PyArrow dataset → Arrow Table ────────────────────────────────
    try:
        arrow_table = ds.dataset(
            str(domain_path),
            format="parquet",
            partitioning="hive",
        ).to_table()
    except Exception as exc:
        logger.error(
            "Cache: failed to read %s from disk: %s", domain, exc, exc_info=True
        )
        return

    read_ms = (time.perf_counter() - t0) * 1000

    # ── 3. Arrow → Polars ────────────────────────────────────────────────────
    t1 = time.perf_counter()
    df = pl.from_arrow(arrow_table)
    del arrow_table  # release Arrow memory; Polars owns the buffer now

    # ── 4. Cast dictionary-encoded columns to plain strings ─────────────────
    #     Do this ONCE at load time so filter comparisons are free on every req.
    cast_exprs = [
        pl.col(name).cast(pl.Utf8)
        for name, dtype in zip(df.columns, df.dtypes)
        if dtype in (pl.Categorical, pl.Enum)
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # ── 5. Store ─────────────────────────────────────────────────────────────
    _DOMAIN_FRAMES[domain] = df

    total_ms = (time.perf_counter() - t0) * 1000
    mem_mb = df.estimated_size("mb")
    _LOAD_STATS[domain] = {
        "rows": len(df),
        "columns": df.width,
        "memory_mb": round(mem_mb, 1),
        "load_time_ms": round(total_ms, 1),
        "read_time_ms": round(read_ms, 1),
    }

    logger.info(
        "Cache: %s loaded  rows=%d  cols=%d  mem=%.1f MB  time=%.0f ms  "
        "(read=%.0f ms  convert=%.0f ms)",
        domain,
        len(df),
        df.width,
        mem_mb,
        total_ms,
        read_ms,
        total_ms - read_ms,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_all_domains(data_root: Path, max_workers: int = 6) -> None:
    """
    Load all SDTM domains from disk into memory, **in parallel**.

    Uses a ``ThreadPoolExecutor`` with up to *max_workers* threads.  I/O-bound
    Parquet reads benefit from parallelism even under the GIL because threads
    are often blocked on kernel I/O, letting other threads run.

    Parameters
    ----------
    data_root:
        Root directory containing ``AE/``, ``CM/``, … sub-folders.
    max_workers:
        Maximum number of parallel reader threads (default 6 = one per domain).
    """
    t0 = time.perf_counter()
    logger.info(
        "Cache: starting parallel load of %d domains (max_workers=%d) …",
        len(DOMAINS),
        max_workers,
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {
            executor.submit(_load_single_domain, domain, data_root): domain
            for domain in DOMAINS
        }
        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            exc = future.exception()
            if exc:
                logger.error("Cache: unhandled error loading %s: %s", domain, exc)

    loaded = sorted(_DOMAIN_FRAMES.keys())
    elapsed = (time.perf_counter() - t0) * 1000
    total_rows = sum(len(df) for df in _DOMAIN_FRAMES.values())
    total_mem = sum(df.estimated_size("mb") for df in _DOMAIN_FRAMES.values())

    logger.info(
        "Cache: all domains loaded  domains=%s  total_rows=%d  "
        "total_memory=%.1f MB  elapsed=%.0f ms",
        loaded,
        total_rows,
        total_mem,
        elapsed,
    )


def get_domain_frame(domain: str) -> pl.DataFrame | None:
    """
    Return the in-memory Polars DataFrame for *domain*, or ``None`` if not
    yet loaded (e.g. the domain data directory was missing at startup).
    """
    return _DOMAIN_FRAMES.get(domain)


def is_loaded(domain: str) -> bool:
    """Return ``True`` if *domain* has been loaded into the cache."""
    return domain in _DOMAIN_FRAMES


def get_cache_stats() -> dict[str, Any]:
    """
    Return a dict of per-domain load statistics suitable for JSON serialisation.

    Useful for the ``/health`` endpoint and operational monitoring.
    """
    total_rows = sum(s["rows"] for s in _LOAD_STATS.values())
    total_mem = sum(s["memory_mb"] for s in _LOAD_STATS.values())
    return {
        "loaded_domains": sorted(_DOMAIN_FRAMES.keys()),
        "total_rows": total_rows,
        "total_memory_mb": round(total_mem, 1),
        "domains": _LOAD_STATS,
    }

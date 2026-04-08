"""
Clinical domain data reader.

Two-tier query strategy
-----------------------
**Tier 1 – In-memory cache (fast path)**
    If ``cache.load_all_domains()`` has been called at application startup
    (which it is via the FastAPI lifespan), every query is served entirely
    from RAM:

    1. Fetch the pre-loaded ``pl.DataFrame`` for the domain.
    2. Build Polars equality predicates and call ``df.lazy().filter(...)``.
    3. Count via ``pl.len()`` on the lazy plan – single-pass.
    4. Slice the lazy frame to the requested page.
    5. ``.collect().to_dicts()`` – vectorised serialisation.

    End-to-end latency: **5–30 ms** even for ``page_size=3 000``.

**Tier 2 – On-disk PyArrow dataset (cold / fallback path)**
    If the domain was not loaded into the cache (missing data directory,
    startup failure, etc.) the original file-based reader is used:

    1. ``pyarrow.dataset.dataset(..., partitioning="hive")`` – file discovery.
    2. ``to_table(filter=expr)`` – pushes equality predicates into Parquet
       row-group statistics; partition-column predicates skip whole directory
       subtrees.
    3. ``pl.from_arrow(table)`` → ``.slice()`` → ``.to_dicts()``.

    End-to-end latency: **3–10 s** (dominated by file-system metadata and
    Parquet decompression).

Filter column mapping
---------------------
The five API query parameters are mapped to the most selective column
available for each domain:

  Parameter  Cache path column   File path column
  ─────────  ──────────────────  ──────────────────────────────────────────
  study      STUDYID             STUDYID  (partition col → pruning)
  site       SITEID *            SITEID * (partition col → pruning)
  subject    USUBJID *           USUBJID * (partition col → pruning)
  visit      VISIT               VISIT    (data col, all domains)
  form       FORM                FORM     (data col, all domains)

  * TV has no SITEID/USUBJID partition.  Those parameters fall back to the
    SITE / SUBJECT data columns present in every TV record.
"""

from __future__ import annotations

import logging
from functools import reduce
from operator import and_
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds

from . import cache as _cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-domain filter-column mappings
# ---------------------------------------------------------------------------
#
# Maps the five API query parameters to the column name used when building
# filter expressions.  Using the partition columns (STUDYID, SITEID, USUBJID)
# for the study / site / subject filters enables:
#   • Cache path: Polars evaluates simple equality scans on Utf8 columns.
#   • File  path: PyArrow skips directory subtrees whose partition value
#                 does not match the predicate (true partition pruning).

_DOMAIN_FILTER_COLUMNS: dict[str, dict[str, str]] = {
    "AE": {
        "study": "STUDYID",
        "site": "SITEID",
        "subject": "USUBJID",
        "visit": "VISIT",
        "form": "FORM",
    },
    "CM": {
        "study": "STUDYID",
        "site": "SITEID",
        "subject": "USUBJID",
        "visit": "VISIT",
        "form": "FORM",
    },
    "DM": {
        "study": "STUDYID",
        "site": "SITEID",
        "subject": "USUBJID",
        "visit": "VISIT",
        "form": "FORM",
    },
    "LB": {
        "study": "STUDYID",
        "site": "SITEID",
        "subject": "USUBJID",
        "visit": "VISIT",
        "form": "FORM",
    },
    # TV is partitioned only by STUDYID; no SITEID/USUBJID partition columns.
    "TV": {
        "study": "STUDYID",
        "site": "SITE",
        "subject": "SUBJECT",
        "visit": "VISIT",
        "form": "FORM",
    },
    "VS": {
        "study": "STUDYID",
        "site": "SITEID",
        "subject": "USUBJID",
        "visit": "VISIT",
        "form": "FORM",
    },
}

_DEFAULT_FILTER_COLUMNS: dict[str, str] = {
    "study": "STUDY",
    "site": "SITE",
    "subject": "SUBJECT",
    "visit": "VISIT",
    "form": "FORM",
}


# ---------------------------------------------------------------------------
# Cache-path helpers  (Polars)
# ---------------------------------------------------------------------------


def _build_polars_predicates(
    domain: str,
    filters: dict[str, str | None],
    available_columns: set[str],
) -> list[pl.Expr]:
    """
    Translate API filter params into Polars equality expressions.

    Only generates predicates for params that have a non-None value *and*
    whose target column actually exists in the DataFrame (guards against
    TV's missing SITEID/USUBJID columns when called with generic mappings).
    """
    col_map = _DOMAIN_FILTER_COLUMNS.get(domain, _DEFAULT_FILTER_COLUMNS)
    predicates: list[pl.Expr] = []

    for param, value in filters.items():
        if value is not None:
            col = col_map.get(param)
            if col and col in available_columns:
                predicates.append(pl.col(col) == value)

    return predicates


def _query_from_cache(
    domain: str,
    df: pl.DataFrame,
    filters: dict[str, str | None],
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Serve a query entirely from the in-memory Polars DataFrame.

    Uses the Polars *lazy* API so that ``filter`` and ``slice`` are fused
    into a single pass by the query planner wherever possible.

    Returns
    -------
    (records, total_matching_rows)
    """
    available = set(df.columns)
    predicates = _build_polars_predicates(domain, filters, available)

    lf: pl.LazyFrame = df.lazy()

    if predicates:
        # Combine multiple predicates with logical AND.
        combined: pl.Expr = reduce(and_, predicates)
        lf = lf.filter(combined)

    # Materialise only the row count first (cheap – uses Polars metadata).
    total: int = lf.select(pl.len()).collect(no_optimization=True).item()

    if total == 0:
        return [], 0

    offset = (page - 1) * page_size
    if offset >= total:
        logger.debug(
            "Cache query: domain=%s page=%d offset=%d >= total=%d – empty page",
            domain,
            page,
            offset,
            total,
        )
        return [], total

    # Slice within the lazy plan so Polars can stop collecting early.
    records: list[dict[str, Any]] = lf.slice(offset, page_size).collect().to_dicts()

    return records, total


# ---------------------------------------------------------------------------
# File-path helpers  (PyArrow)
# ---------------------------------------------------------------------------


def _build_arrow_filter(
    domain: str,
    filters: dict[str, str | None],
) -> ds.Expression | None:
    """
    Translate API filter params into a compound PyArrow dataset expression.

    Returns ``None`` when no filters are active (full-table scan).
    """
    col_map = _DOMAIN_FILTER_COLUMNS.get(domain, _DEFAULT_FILTER_COLUMNS)
    exprs: list[ds.Expression] = []

    for param, value in filters.items():
        if value is not None:
            col = col_map.get(param)
            if col:
                exprs.append(ds.field(col) == value)

    if not exprs:
        return None

    return reduce(and_, exprs)


def _arrow_table_to_polars(table: pa.Table) -> pl.DataFrame:
    """
    Convert a PyArrow Table to a Polars DataFrame.

    Categorical (dictionary-encoded) columns are cast to plain ``Utf8`` so
    that ``to_dicts()`` always yields ordinary Python strings.
    """
    df = pl.from_arrow(table)

    cast_exprs = [
        pl.col(name).cast(pl.Utf8)
        for name, dtype in zip(df.columns, df.dtypes)
        if dtype in (pl.Categorical, pl.Enum)
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def _query_from_files(
    domain_path: Path,
    domain: str,
    filters: dict[str, str | None],
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Fallback: read directly from the Hive-partitioned Parquet tree on disk.

    This is the original slow path (3-10 s per request).  It is only reached
    when the in-memory cache was not populated (e.g. the domain directory was
    missing at startup, or this function is called outside a running server).
    """
    if not domain_path.exists():
        logger.warning(
            "File query: domain directory not found – %s – returning empty result.",
            domain_path,
        )
        return [], 0

    logger.debug(
        "File query (cache MISS): domain=%s path=%s filters=%s",
        domain,
        domain_path,
        {k: v for k, v in filters.items() if v is not None},
    )

    try:
        arrow_dataset: ds.Dataset = ds.dataset(
            str(domain_path),
            format="parquet",
            partitioning="hive",
        )
    except Exception as exc:
        logger.error("File query: failed to open dataset domain=%s: %s", domain, exc)
        return [], 0

    arrow_filter = _build_arrow_filter(domain, filters)

    try:
        arrow_table: pa.Table = arrow_dataset.to_table(filter=arrow_filter)
    except Exception as exc:
        logger.error(
            "File query: scan failed domain=%s: %s", domain, exc, exc_info=True
        )
        return [], 0

    total: int = arrow_table.num_rows
    if total == 0:
        return [], 0

    df = _arrow_table_to_polars(arrow_table)
    del arrow_table  # release Arrow memory

    offset = (page - 1) * page_size
    if offset >= total:
        return [], total

    records = df.slice(offset, page_size).to_dicts()
    return records, total


# ---------------------------------------------------------------------------
# Public query entry point
# ---------------------------------------------------------------------------


def query_domain(
    domain_path: Path,
    domain: str,
    filters: dict[str, str | None],
    page: int,
    page_size: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Read, filter, and paginate records from a single SDTM domain.

    Automatically selects the **cache path** (Polars in-memory, ~5–30 ms) or
    the **file path** (PyArrow on-disk, ~3–10 s) depending on whether the
    domain has been loaded into the module-level cache.

    Parameters
    ----------
    domain_path:
        Absolute path to the domain root, e.g. ``…/clinical_data_output/AE``.
        Used only by the file-path fallback; ignored when cache is warm.
    domain:
        SDTM domain code – ``"AE"``, ``"CM"``, ``"DM"``, ``"LB"``, ``"TV"``,
        or ``"VS"``.
    filters:
        Mapping of API parameter name → string value or ``None`` (= no filter).
    page:
        1-indexed page number.
    page_size:
        Maximum records per page (1 – 3 000).

    Returns
    -------
    tuple[list[dict], int]
        ``(records, total_matching_records)``

        *records* – list of plain Python dicts ready for JSON serialisation.
        Date fields are ``datetime.date`` instances; all string fields are
        ``str``; numeric fields are ``int`` or ``float``; missing values are
        ``None``.

        *total_matching_records* – total count **before** pagination, used to
        compute ``total_pages`` in the response meta block.
    """
    cached_df = _cache.get_domain_frame(domain)

    if cached_df is not None:
        active = {k: v for k, v in filters.items() if v is not None}
        logger.debug(
            "Cache HIT: domain=%s  filters=%s  page=%d  page_size=%d",
            domain,
            active,
            page,
            page_size,
        )
        return _query_from_cache(domain, cached_df, filters, page, page_size)

    # Cache miss – fall back to reading from disk.
    logger.warning(
        "Cache MISS for domain=%s – falling back to disk read (slow). "
        "Was load_all_domains() called at startup?",
        domain,
    )
    return _query_from_files(domain_path, domain, filters, page, page_size)


# ---------------------------------------------------------------------------
# Convenience: count-only query
# ---------------------------------------------------------------------------


def count_domain(
    domain_path: Path,
    domain: str,
    filters: dict[str, str | None] | None = None,
) -> int:
    """
    Return the number of records matching *filters* without fetching any rows.

    Uses the in-memory cache when available (very fast).  Falls back to a
    PyArrow ``scanner(columns=[])`` count when the cache is cold (reads only
    row-group metadata, no column data decoded).
    """
    effective_filters: dict[str, str | None] = filters or {}

    cached_df = _cache.get_domain_frame(domain)
    if cached_df is not None:
        available = set(cached_df.columns)
        predicates = _build_polars_predicates(domain, effective_filters, available)
        if not predicates:
            return len(cached_df)
        combined: pl.Expr = reduce(and_, predicates)
        return cached_df.filter(combined).height

    # Fallback: PyArrow metadata-only count.
    if not domain_path.exists():
        return 0

    try:
        arrow_dataset = ds.dataset(
            str(domain_path),
            format="parquet",
            partitioning="hive",
        )
        arrow_filter = _build_arrow_filter(domain, effective_filters)
        return arrow_dataset.scanner(filter=arrow_filter, columns=[]).count_rows()
    except Exception as exc:
        logger.error("count_domain failed domain=%s: %s", domain, exc)
        return 0

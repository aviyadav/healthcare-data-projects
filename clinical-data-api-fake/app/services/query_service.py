"""
Core query service.

Responsible for:
1. Building parameterized SQL WHERE clauses from common filter params
2. Executing queries via DuckDB, fetching as Apache Arrow tables
3. Converting Arrow tables → Polars LazyFrames for zero-copy slicing
4. Returning paginated JSON-serializable dicts

Design decisions
----------------
- DuckDB → Arrow: uses `.fetch_arrow_table()` which performs a true zero-copy
  transfer of columnar data into Arrow format.
- Arrow → Polars: `pl.from_arrow()` wraps the Arrow table without copying memory.
  Polars LazyFrame is used so we can chain operations (filter/slice) without
  materialising the entire dataset.
- Pagination is done in SQL (LIMIT/OFFSET) so the DB does the heavy lifting and
  only the requested page is transferred over the Python boundary.
- COUNT(*) is run as a separate lightweight query so we can return total_records
  without fetching all data.
- Blocking DuckDB I/O is offloaded to a thread-pool via asyncio.to_thread so the
  async event loop is never stalled while the database is working.
"""
import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from fastapi import HTTPException, status

from app.database import get_db
from app.models.schemas import PaginationMeta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common filter parameters definition
# ---------------------------------------------------------------------------

# Columns shared by every domain table that can be filtered
COMMON_FILTER_COLUMNS = ("STUDY", "SITE", "SUBJECT", "VISIT", "FORM")


def build_filter_clause(
    study: Optional[str] = None,
    site: Optional[str] = None,
    subject: Optional[str] = None,
    visit: Optional[str] = None,
    form: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """
    Build a parameterized SQL WHERE clause from the provided filters.

    Returns
    -------
    (where_clause, params)
        where_clause : str  — e.g. "WHERE STUDY = ? AND SITE = ?"
        params       : list — positional parameter values for DuckDB
    """
    conditions: List[str] = []
    params: List[Any] = []

    filter_map = {
        "STUDY": study,
        "SITE": site,
        "SUBJECT": subject,
        "VISIT": visit,
        "FORM": form,
    }

    for col, value in filter_map.items():
        if value is not None:
            conditions.append(f"{col} = ?")
            params.append(value)

    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return where_clause, params


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------


async def fetch_domain_data(
    table: str,
    study: Optional[str] = None,
    site: Optional[str] = None,
    subject: Optional[str] = None,
    visit: Optional[str] = None,
    form: Optional[str] = None,
    page: int = 1,
    page_size: int = 100,
) -> Tuple[List[Dict[str, Any]], PaginationMeta]:
    """
    Fetch a paginated, filtered slice of a domain table.

    Parameters
    ----------
    table      : SDTM domain table name (e.g. "AE", "CM")
    study      : optional STUDY filter
    site       : optional SITE filter
    subject    : optional SUBJECT filter
    visit      : optional VISIT filter
    form       : optional FORM filter
    page       : 1-indexed page number
    page_size  : records per page

    Returns
    -------
    (records, meta)
        records : list of dicts (JSON-serialisable)
        meta    : PaginationMeta instance
    """
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="page must be >= 1",
        )

    where_clause, params = build_filter_clause(study, site, subject, visit, form)
    offset = (page - 1) * page_size

    count_sql = f"SELECT COUNT(*) AS cnt FROM {table} {where_clause}"
    data_sql = (
        f"SELECT * FROM {table} {where_clause} "
        f"LIMIT ? OFFSET ?"
    )

    def _execute_queries() -> Tuple[int, Any]:
        """Run both DB queries synchronously inside a worker thread."""
        with get_db() as conn:
            # -- Count query (fast, full-table stats)
            try:
                total: int = conn.execute(count_sql, params).fetchone()[0]  # type: ignore[index]
            except Exception as exc:
                logger.error("Count query failed for table %s: %s", table, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error fetching count from {table}.",
                ) from exc

            # -- Data query via Arrow → Polars (zero-copy columnar path)
            try:
                arrow = conn.execute(data_sql, params + [page_size, offset]).to_arrow_table()
            except Exception as exc:
                logger.error("Data query failed for table %s: %s", table, exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error fetching data from {table}.",
                ) from exc

            return total, arrow

    # Offload blocking DB I/O to a thread so the event loop stays free
    total_records, arrow_table = await asyncio.to_thread(_execute_queries)

    # Convert Arrow → Polars (zero-copy)
    if arrow_table.num_rows == 0:
        records: List[Dict[str, Any]] = []
    else:
        df: pl.DataFrame = pl.from_arrow(arrow_table)
        # Cast date columns from Polars Date to Python date strings for JSON
        # Polars handles date serialisation via to_dicts() natively
        records = _polars_to_records(df)

    total_pages = max(1, math.ceil(total_records / page_size)) if total_records > 0 else 0

    meta = PaginationMeta(
        page=page,
        page_size=page_size,
        total_records=total_records,
        total_pages=total_pages,
    )

    return records, meta


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _polars_to_records(df: pl.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a Polars DataFrame to a list of JSON-serialisable dicts.

    Date columns are converted to ISO-8601 strings. All other types are passed
    through as-is (Polars handles int/float/str natively in to_dicts()).
    """
    # Cast Polars Date → Utf8 so Python `date` objects become ISO strings
    cast_exprs = [
        pl.col(c).cast(pl.Utf8).alias(c)
        for c in df.columns
        if df[c].dtype == pl.Date
    ]

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df.to_dicts()

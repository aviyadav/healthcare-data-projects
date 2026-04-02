# Clinical Data FastAPI — Implementation Plan

## Overview

Build a production-grade async FastAPI application that serves clinical trial data from a DuckDB database. The API will expose six domain endpoints (AE, CM, DM, LB, TV, VS), support filtering, pagination, and use Polars + PyArrow for efficient columnar data processing. Gunicorn with Uvicorn workers will provide multiprocessing concurrency.

---

## Proposed Project Structure

```
clinical-data-api-fake/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app entrypoint
│   ├── config.py             # Settings (DB path, pagination defaults, etc.)
│   ├── database.py           # DuckDB connection pool manager
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic response models for each domain
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ae.py             # Adverse Events endpoint
│   │   ├── cm.py             # Concomitant Medications endpoint
│   │   ├── dm.py             # Demographics endpoint
│   │   ├── lb.py             # Laboratory endpoint
│   │   ├── tv.py             # Trial Visits endpoint
│   │   └── vs.py             # Vital Signs endpoint
│   └── services/
│       ├── __init__.py
│       └── query_service.py  # Core query logic (DuckDB → Polars → PyArrow → JSON)
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   └── test_api.py           # API endpoint tests
├── main.py                   # Entry point (replaces old main.py)
├── pyproject.toml            # Updated with all dependencies
├── gunicorn.conf.py          # Gunicorn worker config
└── README.md                 # Updated with setup/run/test instructions
```

---

## Proposed Changes

### Dependencies

#### [MODIFY] [pyproject.toml](file:///home/avinash/codebase/python-base/healthcare-data-projects/clinical-data-api-fake/pyproject.toml)

Add:
- `fastapi`, `uvicorn[standard]`, `gunicorn` — web server stack
- `duckdb` — database driver
- `polars`, `pyarrow` — fast columnar data processing
- `pydantic`, `pydantic-settings` — data validation & settings
- `pytest`, `pytest-asyncio`, `httpx` — testing

---

### Core Application

#### [NEW] app/config.py
- `Settings` class using `pydantic-settings`
- Configurable: `DB_PATH`, `MAX_WORKERS`, `DEFAULT_PAGE_SIZE`, `MAX_PAGE_SIZE`

#### [NEW] app/database.py
- Thread-local DuckDB connections (DuckDB supports read-only multi-process)
- Context manager for safe connection lifecycle
- Opens DB in **read-only** mode to allow multiple concurrent processes

#### [NEW] app/services/query_service.py
- `build_filter_clause()` — builds parameterized SQL WHERE clause from common filters
- `execute_query()` — runs DuckDB query, converts result to Polars DataFrame via PyArrow, applies pagination
- `df_to_json_response()` — serializes Polars DataFrame to list of dicts efficiently

#### [NEW] app/models/schemas.py
- `PaginationMeta` — page, page_size, total_records, total_pages
- `PaginatedResponse[T]` — generic wrapper with data + meta
- Per-domain Pydantic models: `AERecord`, `CMRecord`, `DMRecord`, `LBRecord`, `TVRecord`, `VSRecord`

---

### Routers (one per domain)

Each router follows the same pattern:

```
GET /api/v1/{domain}?study=&site=&subject=&visit=&form=&page=1&page_size=100
```

Returns:
```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "page_size": 100,
    "total_records": 5000,
    "total_pages": 50
  }
}
```

#### [NEW] app/routers/ae.py — `GET /api/v1/ae`
#### [NEW] app/routers/cm.py — `GET /api/v1/cm`
#### [NEW] app/routers/dm.py — `GET /api/v1/dm`
#### [NEW] app/routers/lb.py — `GET /api/v1/lb`
#### [NEW] app/routers/tv.py — `GET /api/v1/tv`
#### [NEW] app/routers/vs.py — `GET /api/v1/vs`

---

### Main App

#### [MODIFY] app/main.py
- FastAPI app with title, version, description
- Register all domain routers under `/api/v1`
- Add `/health` endpoint
- Add `/api/v1/domains` metadata endpoint listing available domains

#### [MODIFY] main.py (root)
- Simple entrypoint: `uvicorn app.main:app`

#### [NEW] gunicorn.conf.py
- `workers = multiprocessing.cpu_count() * 2 + 1`
- `worker_class = "uvicorn.workers.UvicornWorker"`
- Bind to `0.0.0.0:8000`

---

### Tests

#### [NEW] tests/conftest.py
- `TestClient` fixture pointing at the app

#### [NEW] tests/test_api.py
- Test `/health` 
- Test each domain endpoint returns 200 with pagination meta
- Test filter parameters (study, site, subject, visit, form)
- Test pagination boundaries (page=1, page_size=10)
- Test invalid filter returns empty data (not error)

---

### Documentation

#### [MODIFY] README.md
- Project overview
- Prerequisites (Python 3.13+, uv)
- Installation steps
- Running with uvicorn (dev)
- Running with gunicorn (production)
- API reference (all endpoints + query params)
- Running tests

---

## Verification Plan

### Automated Tests
```bash
uv run pytest tests/ -v
```

### Manual Verification
- Start server and hit each endpoint via browser/curl
- Confirm pagination metadata is accurate
- Confirm filter parameters reduce result set correctly


# Clinical Data API

A high-performance **FastAPI** REST service that exposes synthetically generated
CDISC SDTM clinical trial data across six domains.

| Code | Domain | Description |
|------|--------|-------------|
| **AE** | Adverse Events | Undesirable medical occurrences during the trial |
| **CM** | Concomitant Medications | Drugs taken alongside the study treatment |
| **DM** | Demographics | Baseline characteristics of each study subject |
| **LB** | Laboratory Results | Clinical lab test values and reference ranges |
| **TV** | Trial Visits | Planned visit schedule and window definitions |
| **VS** | Vital Signs | Blood pressure, heart rate, temperature, weight, etc. |

---

## Technology Stack

| Layer | Library | Role |
|-------|---------|------|
| **API framework** | [FastAPI](https://fastapi.tiangolo.com/) ≥ 0.115 | Async REST API, OpenAPI docs, Pydantic validation |
| **Production server** | [Gunicorn](https://gunicorn.org/) ≥ 22 + [Uvicorn](https://www.uvicorn.org/) workers | Multi-process ASGI server with process supervision |
| **Schema registry** | [PyIceberg](https://py.iceberg.apache.org/) ≥ 0.7 | SDTM domain schemas, partition specs, SQLite-backed catalog |
| **Data I/O** | [PyArrow](https://arrow.apache.org/docs/python/) ≥ 18 | Hive-partitioned Parquet discovery, filter pushdown, zero-copy Arrow tables |
| **In-memory engine** | [Polars](https://docs.pola.rs/) ≥ 1.14 | Vectorised in-memory filtering, lazy query planning, pagination |
| **Serialisation** | [orjson](https://github.com/ijl/orjson) ≥ 3.10 | 5-10× faster JSON encoding; native `datetime.date` → ISO-8601 |

---

## Architecture

### Request flow

```
 Startup (once per worker)
 ┌────────────────────────────────────────────────────────────┐
 │  PyArrow  ds.dataset(…, partitioning="hive").to_table()    │
 │  → pl.from_arrow()  → cast Categorical → Utf8              │
 │  → store 6 × pl.DataFrame in RAM  (~153 MB total)          │
 └────────────────────────────────────────────────────────────┘

 Every request  (~10–65 ms end-to-end)
 ┌────────────────────────────────────────────────────────────┐
 │  1. PyIceberg SqlCatalog  (SQLite)                         │
 │     Powers /api/v1/domains metadata endpoint only.         │
 │     Schema & partition-spec objects always available        │
 │     from in-memory DOMAIN_SCHEMAS dict.                     │
 └───────────────────────┬────────────────────────────────────┘
                         │ schema introspection only
                         ▼
 ┌────────────────────────────────────────────────────────────┐
 │  In-memory cache  (app/data/cache.py)                      │
 │     _DOMAIN_FRAMES["AE" | "CM" | …]  →  pl.DataFrame       │
 └───────────────────────┬────────────────────────────────────┘
                         │ get_domain_frame(domain)
                         ▼
 ┌────────────────────────────────────────────────────────────┐
 │  Polars lazy query  (app/data/reader.py)                   │
 │     df.lazy()                                              │
 │       .filter(pl.col("STUDYID") == study & …)              │
 │       .slice(offset, page_size)                            │
 │       .collect()                                           │
 │       .to_dicts()                                          │
 └───────────────────────┬────────────────────────────────────┘
                         │ list[dict]
                         ▼
 ┌────────────────────────────────────────────────────────────┐
 │  orjson.dumps()  →  Response(media_type="application/json")│
 │  (bypasses Pydantic per-record output validation)          │
 └────────────────────────────────────────────────────────────┘
```

### Fallback path (cache miss)

If a domain was not loaded at startup (missing directory, I/O error), the reader
falls back to the original on-disk path:

```
PyArrow ds.dataset() → to_table(filter=expr) → pl.from_arrow() → slice → to_dicts()
```

This is ~100-200× slower but keeps the API available during partial failures.

---

## Project Structure

```
clinical_data_api/
├── gunicorn.conf.py          # Gunicorn configuration (workers, timeouts, hooks)
├── requirements.txt
├── README.md
└── app/
    ├── main.py               # FastAPI app, lifespan (cache load + catalog init)
    ├── config.py             # Pydantic-Settings with CLINICAL_ env var overrides
    ├── models.py             # Pydantic response models (AERecord, CMRecord, …)
    ├── data/
    │   ├── cache.py          # In-memory domain cache (parallel load at startup)
    │   ├── iceberg_schemas.py  # PyIceberg Schema + PartitionSpec for all 6 domains
    │   ├── catalog.py        # SqlCatalog bootstrap & list_domain_metadata()
    │   └── reader.py         # Two-tier reader: cache → file fallback
    └── routers/
        ├── ae.py   → GET /api/v1/ae
        ├── cm.py   → GET /api/v1/cm
        ├── dm.py   → GET /api/v1/dm
        ├── lb.py   → GET /api/v1/lb
        ├── tv.py   → GET /api/v1/tv
        ├── vs.py   → GET /api/v1/vs
        └── system.py  → GET /health  +  GET /api/v1/domains
```

---

## Data Layout

```
clinical_data_output/
├── AE/  STUDYID=…/ SITEID=…/ USUBJID=…/ AE_INCIDENT_GROUP=…/ part-0-*.parquet
├── CM/  STUDYID=…/ SITEID=…/ USUBJID=…/ part-0-*.parquet
├── DM/  STUDYID=…/ SITEID=…/ USUBJID=…/ part-0-*.parquet
├── LB/  STUDYID=…/ SITEID=…/ USUBJID=…/ part-0-*.parquet
├── TV/  STUDYID=…/ part-0-*.parquet          ← no SITEID / USUBJID partition
└── VS/  STUDYID=…/ SITEID=…/ USUBJID=…/ part-0-*.parquet
```

Partition column values (`STUDYID`, `SITEID`, `USUBJID`, `AE_INCIDENT_GROUP`) are
encoded in both the directory path **and** as dictionary-typed columns inside each
Parquet file.  PyArrow's `partitioning="hive"` resolves them from the path, so no
duplicate columns appear in the loaded DataFrames.

---

## Installation

```bash
cd clinical_data_api
pip install -r requirements.txt
```

---

## Running the Server

> **Windows users:** Gunicorn uses POSIX-only syscalls (`fcntl`, `os.fork`) and
> **will not run on Windows**.  Use `start_server.py` or `start_server.bat`
> (described below) — they auto-detect the platform and choose the right server.

### Recommended — cross-platform launcher (all platforms)

The included `start_server.py` script automatically selects the correct server:

| Platform | Server used | Notes |
|----------|-------------|-------|
| Linux / macOS / WSL | Gunicorn + UvicornWorker | Full process supervision, OS copy-on-write memory sharing |
| Windows | Uvicorn `--workers` | Each worker loads data independently; memory is **not** shared |

**Python / CMD / PowerShell:**

```bash
# From the clinical_data_api/ directory
python start_server.py
```

**Windows batch file (double-click or from CMD):**

```bat
start_server.bat
```

**Override worker count or bind address:**

```bash
# Linux / macOS / WSL
CLINICAL_WORKERS=4 python start_server.py

# Windows CMD
set CLINICAL_WORKERS=4 && python start_server.py

# Windows PowerShell
$env:CLINICAL_WORKERS=4; python start_server.py
```

---

### Development — single process with auto-reload
</thinking>

```bash
# From the clinical_data_api/ directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090


uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8090
```

> The `--reload` flag restarts the worker on code changes.
> Do **not** use `--reload` in production — it disables the in-memory cache
> between reloads and prevents Gunicorn's process supervision.

Open **http://localhost:8090/docs** for the interactive Swagger UI.

---

### Production — Gunicorn with Uvicorn workers (Linux / macOS / WSL)

Gunicorn supervises a pool of Uvicorn worker processes.  Each worker loads
the full dataset into RAM independently; on Linux the OS copy-on-write
mechanism keeps the Arrow column buffers physically shared across workers as
long as they remain read-only.

#### Default configuration (auto-detects CPU count)

```bash
# From the clinical_data_api/ directory
gunicorn app.main:app -c gunicorn.conf.py
```

This starts `(2 × CPU_COUNT) + 1` workers bound to `0.0.0.0:8090`.

#### Explicit worker count

```bash
# 4 workers
gunicorn app.main:app -c gunicorn.conf.py -w 4

# Override via environment variable (takes precedence over -w)
CLINICAL_WORKERS=8 gunicorn app.main:app -c gunicorn.conf.py
```

#### Custom bind address / port

```bash
# Bind to a specific interface and port
gunicorn app.main:app -c gunicorn.conf.py --bind 127.0.0.1:9000

# Or via env var in the config
CLINICAL_BIND=0.0.0.0:9000 gunicorn app.main:app -c gunicorn.conf.py
```

#### Run in the background (daemon mode)

```bash
gunicorn app.main:app -c gunicorn.conf.py \
  --daemon \
  --pid /var/run/clinical-api.pid \
  --access-logfile /var/log/clinical-api-access.log \
  --error-logfile  /var/log/clinical-api-error.log
```

#### Graceful reload (zero-downtime restart)

```bash
# Send HUP to the master process to reload workers without dropping connections
kill -HUP $(cat /var/run/clinical-api.pid)
```

#### Stop the server

```bash
# Graceful shutdown (waits for in-flight requests)
kill -TERM $(cat /var/run/clinical-api.pid)

# Immediate shutdown
kill -INT $(cat /var/run/clinical-api.pid)
```

---

### Production — Uvicorn with workers (Windows / cross-platform)

> **Why not Gunicorn on Windows?**
> Gunicorn imports `fcntl` and calls `os.fork()` — both are POSIX-only and do
> not exist on Windows.  Attempting to run it will immediately raise
> `ModuleNotFoundError: No module named 'fcntl'`.  Use `start_server.py` or
> the raw Uvicorn command below instead.

Uvicorn's built-in multi-process mode works on all platforms:

```bash
# From the clinical_data_api/ directory
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8090
```

> On Windows, Python uses `spawn` (not `fork`) for new processes, so each
> worker starts independently and loads its own copy of the dataset.
> Memory is **not** shared between workers on Windows — plan for
> approximately `N_workers × dataset_size` RAM.

---

### Launcher configuration reference (`start_server.py` / `gunicorn.conf.py`)

`start_server.py` reads the same environment variables as `gunicorn.conf.py`,
so a single `.env` file or `set` command works on every platform.

### Gunicorn configuration reference (`gunicorn.conf.py`)

| Setting | Default | Override | Description |
|---------|---------|----------|-------------|
| `bind` | `0.0.0.0:8090` | `CLINICAL_BIND` | Host and port to listen on |
| `workers` | `(2 × CPU) + 1` | `CLINICAL_WORKERS` | Number of worker processes |
| `worker_class` | `uvicorn.workers.UvicornWorker` | — | ASGI worker implementation |
| `timeout` | `120` | `CLINICAL_TIMEOUT` | Worker silence timeout (seconds) |
| `preload_app` | `True` | — | Import app once in master before forking |
| `worker_tmp_dir` | `/dev/shm` (Linux) | — | In-memory worker heartbeat files |
| `loglevel` | `info` | `CLINICAL_LOG_LEVEL` | Logging verbosity |

---

## Environment Variables

All settings in `app/config.py` accept `CLINICAL_` prefixed environment
variables or a `.env` file in the working directory.

| Variable | Default | Description |
|----------|---------|-------------|
| `CLINICAL_DATA_ROOT` | `../clinical_data_output` | Path to the Parquet data tree |
| `CLINICAL_CATALOG_URI` | `sqlite:///…/clinical_catalog.db` | PyIceberg catalog SQLite URI |
| `CLINICAL_DEBUG` | `false` | Enable DEBUG-level logging |
| `CLINICAL_WORKERS` | `(2 × CPU) + 1` | Gunicorn worker count |
| `CLINICAL_BIND` | `0.0.0.0:8090` | Gunicorn bind address |
| `CLINICAL_TIMEOUT` | `120` | Gunicorn worker timeout (seconds) |
| `CLINICAL_LOG_LEVEL` | `info` | Gunicorn log level |

Example `.env` file:

```ini
CLINICAL_DATA_ROOT=/data/clinical_data_output
CLINICAL_WORKERS=8
CLINICAL_BIND=0.0.0.0:8090
CLINICAL_LOG_LEVEL=warning
CLINICAL_DEBUG=false
```

---

## API Reference

### Common conventions

* **Pagination** – `page` (1-indexed, default `1`) and `page_size` (1–3000, default `1000`).
* **Filters** – all five filter parameters are optional; any combination may be supplied.
* **Response envelope** – every list endpoint returns:

```json
{
  "data": [ { "STUDY": "STUDY-001", "SITE": "SITE-001", … }, … ],
  "meta": {
    "page": 1,
    "page_size": 1000,
    "total_records": 203418,
    "total_pages": 204
  }
}
```

### Domain endpoints

| Method | Path | Domain |
|--------|------|--------|
| `GET` | `/api/v1/ae` | Adverse Events |
| `GET` | `/api/v1/cm` | Concomitant Medications |
| `GET` | `/api/v1/dm` | Demographics |
| `GET` | `/api/v1/lb` | Laboratory Results |
| `GET` | `/api/v1/tv` | Trial Visits |
| `GET` | `/api/v1/vs` | Vital Signs |

#### Common query parameters (all optional)

| Parameter | Type | Example | Description |
|-----------|------|---------|-------------|
| `study` | string | `STUDY-001` | Filter by STUDY identifier |
| `site` | string | `SITE-003` | Filter by SITE identifier |
| `subject` | string | `STUDY-001-SITE-001-00efc4b2` | Filter by subject / USUBJID |
| `visit` | string | `SCREENING` | Filter by visit name |
| `form` | string | `AE` | Filter by CRF form name |
| `page` | integer | `2` | Page number (≥ 1, default `1`) |
| `page_size` | integer | `500` | Records per page (1–3000, default `1000`) |

### System endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe; includes in-memory cache statistics |
| `GET` | `/api/v1/domains` | All domain schemas and partition specs from PyIceberg |

---

## Example Requests

```bash
# First page of all adverse events (default page_size=1000)
curl "http://localhost:8090/api/v1/ae"

# AEs for a specific study and site, max page size
curl "http://localhost:8090/api/v1/ae?study=STUDY-001&site=SITE-002&page_size=3000"

# Demographics for a specific subject
curl "http://localhost:8090/api/v1/dm?subject=STUDY-001-SITE-001-00efc4b2"

# Lab results – page 3, 500 records per page
curl "http://localhost:8090/api/v1/lb?study=STUDY-002&page=3&page_size=500"

# Vital signs filtered by visit
curl "http://localhost:8090/api/v1/vs?study=STUDY-001&visit=WEEK%208"

# Domain schemas and partition metadata from PyIceberg catalog
curl "http://localhost:8090/api/v1/domains"

# Health check including cache stats
curl "http://localhost:8090/health"
```

---

## Performance

### Startup

At startup the server loads all six SDTM domains from Hive-partitioned Parquet
into process-memory Polars DataFrames **in parallel** (one thread per domain).

| Domain | Rows | Memory |
|--------|------|--------|
| AE | 37,763 | 6.8 MB |
| CM | 12,604 | 2.1 MB |
| DM | 5,000 | 0.6 MB |
| LB | 406,728 | 57.3 MB |
| TV | 208 | < 0.1 MB |
| VS | 677,880 | 86.5 MB |
| **Total** | **1,140,183** | **153.3 MB** |

Cold-start load time: ~30 s on a typical developer laptop (parallel I/O over
a local filesystem with 8 000+ small Parquet files).  On NVMe storage or a
network filesystem the load is proportionally faster.

### Per-request latency (cache warm, `page_size=3000`)

These figures were measured with the TestClient on a Windows laptop.
Real-world server-side latency will be similar or lower on Linux.

| Endpoint | Matching rows | Query time | Serialise | Total |
|----------|--------------|-----------|-----------|-------|
| `GET /api/v1/ae?study=STUDY-001&site=SITE-001` | 3,931 | 145 ms | 4 ms | **64 ms** |
| `GET /api/v1/cm?study=STUDY-001` | 6,262 | 27 ms | 4 ms | **50 ms** |
| `GET /api/v1/dm?study=STUDY-001` | 2,501 | 16 ms | 3 ms | **35 ms** |
| `GET /api/v1/lb?study=STUDY-002` | 203,418 | 17 ms | 4 ms | **37 ms** |
| `GET /api/v1/tv?study=STUDY-001` | 104 | 3 ms | < 1 ms | **10 ms** |
| `GET /api/v1/vs?study=STUDY-001&site=SITE-002` | 65,250 | 18 ms | 3 ms | **52 ms** |

**Before caching: 5,000–10,000 ms.  After caching: 10–65 ms.  ~175× speedup.**

### Why it is fast

| Technique | Benefit |
|-----------|---------|
| **In-memory cache** | Eliminates all file I/O on the hot path (was 99% of request time) |
| **Parallel load** | 6 domains loaded concurrently; wall-clock startup time ≈ slowest single domain |
| **Polars lazy API** | `filter` + `slice` are fused by the query planner into a single pass |
| **`orjson`** | 5-10× faster JSON encoding than stdlib `json`; native `datetime.date` support |
| **Skip Pydantic output validation** | Routers return `Response(orjson.dumps(…))` directly — no per-record Pydantic model instantiation on the response path |
| **Gunicorn workers** | Multiple processes handle concurrent requests; each worker serves from its own in-memory cache |
| **`/dev/shm` worker tmp** | Gunicorn liveness heartbeats use tmpfs on Linux — no disk I/O |

---

## PyIceberg Catalog

At startup the application initialises a **SQLite-backed `SqlCatalog`** as a
schema registry.  It carries type information and partition specs for all six
domains but holds no data files.

```
clinical_catalog.db
  └── namespace: clinical
        ├── clinical.AE   19 fields  partitioned by STUDYID / SITEID / USUBJID / AE_INCIDENT_GROUP
        ├── clinical.CM   20 fields  partitioned by STUDYID / SITEID / USUBJID
        ├── clinical.DM   15 fields  partitioned by STUDYID / SITEID / USUBJID
        ├── clinical.LB   16 fields  partitioned by STUDYID / SITEID / USUBJID
        ├── clinical.TV   11 fields  partitioned by STUDYID
        └── clinical.VS   14 fields  partitioned by STUDYID / SITEID / USUBJID
```

> **Windows note** — PyIceberg's `SqlCatalog` warehouse write uses
> `urllib.parse.urlparse` internally and misinterprets Windows drive letters
> (e.g. `C:`) as URI schemes, so table-metadata JSON files cannot be
> persisted to the warehouse directory on Windows.  This is non-fatal: the
> `/api/v1/domains` endpoint reads schemas directly from the in-memory
> `DOMAIN_SCHEMAS` PyIceberg objects and is always available.  The issue does
> not occur on Linux / macOS.

---

## Smoke Test & Benchmark

A self-contained test script is included.  It runs four sections in sequence:
catalog metadata, cache load timing, hot-path query benchmark, and live
endpoint checks via FastAPI's `TestClient`.

```bash
cd clinical_data_api
python test_smoke.py
```

Expected output (abbreviated):

```
1. PyIceberg catalog
  AE    fields=19  partitions=['STUDYID', 'SITEID', 'USUBJID', 'AE_INCIDENT_GROUP']
  …
  ✓ catalog OK

2. In-memory cache load (PyArrow → Polars, parallel)
  Loaded 6 domains in 29884 ms
  Total rows:   1,140,183
  Total memory: 153.3 MB
  ✓ cache OK

3. Hot-path query benchmark (cache warm, page_size=3000)
  LB    total= 203,418  page=3,000  query=  17.2 ms  serial=  4.4 ms
  ✓ benchmark OK

4. FastAPI endpoints via TestClient (lifespan = warm cache)
  GET /api/v1/lb       → 200  total= 203,418  page=3,000  time=   37.0 ms
  ✓ endpoints OK

ALL SMOKE TESTS PASSED
```

---

## Deployment Checklist

- [ ] Set `CLINICAL_DATA_ROOT` to the absolute path of the Parquet tree
- [ ] Set `CLINICAL_WORKERS` to `(2 × vCPU) + 1` (or tune per load test)
- [ ] Ensure the working directory is `clinical_data_api/` when starting Gunicorn
- [ ] Mount the data directory as **read-only** in containers (`ro` flag)
- [ ] Set `CLINICAL_LOG_LEVEL=warning` in production to reduce log volume
- [ ] Point a reverse proxy (nginx / ALB) at `0.0.0.0:8090`
- [ ] Configure the reverse proxy read timeout ≥ 120 s (covers cold-start load)
- [ ] Add a liveness probe to `GET /health` (returns 200 once cache is warm)
- [ ] Add a readiness probe that checks `cache.loaded_domains` length == 6

---

## License

This project uses synthetically generated data for demonstration purposes only.
No real patient data is included.

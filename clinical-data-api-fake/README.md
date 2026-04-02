# Clinical Data API

A high-performance asynchronous REST API for accessing clinical trial data across CDISC SDTM domains (AE, CM, DM, LB, TV, VS).

Powered by:
- **FastAPI** for async routing and validation
- **DuckDB** for ultra-fast analytical queries over the clinical database
- **Polars & PyArrow** for zero-copy columnar data slicing and processing
- **Gunicorn & Uvicorn** for robust multiprocessing

## Prerequisites
- Python 3.13+
- `uv` (for dependency management)

## Setup

1. Install dependencies:
   ```bash
   uv sync --extra dev
   ```
2. Ensure the DuckDB database file `clinical_data.duckdb` is present in the project root.

## Running

### Development Mode
Runs the API with hot-reloading:
```bash
uv run python main.py
```
Or directly with uvicorn:
```bash
uv run uvicorn app.main:app --port 8090 --reload
```

### Production Mode

**Linux / macOS** — Gunicorn spawns Uvicorn workers across all CPU cores:
```bash
uv run gunicorn -c gunicorn.conf.py app.main:app
```

**Windows** — Gunicorn requires Unix (`fcntl`) and does not run on Windows. Use Uvicorn's built-in multi-worker mode instead:
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8090 --workers 4
```
Set `--workers` to `(2 × CPU cores) + 1` for best throughput.

## API Reference

Interactive documentation is auto-generated and available once the server is running:
- Swagger UI: [http://localhost:8090/docs](http://localhost:8090/docs)
- ReDoc: [http://localhost:8090/redoc](http://localhost:8090/redoc)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/api/v1/domains` | List all available SDTM domains |
| `GET` | `/api/v1/ae` | Adverse Events |
| `GET` | `/api/v1/cm` | Concomitant Medications |
| `GET` | `/api/v1/dm` | Demographics |
| `GET` | `/api/v1/lb` | Laboratory Results |
| `GET` | `/api/v1/tv` | Trial Visits |
| `GET` | `/api/v1/vs` | Vital Signs |

### Query Parameters

All domain endpoints accept the following optional query parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `study` | string | — | Filter by STUDY identifier |
| `site` | string | — | Filter by SITE identifier |
| `subject` | string | — | Filter by SUBJECT identifier |
| `visit` | string | — | Filter by VISIT name |
| `form` | string | — | Filter by FORM name |
| `page` | int ≥ 1 | `1` | Page number (1-indexed) |
| `page_size` | int 1–1000 | `100` | Records per page |

### Response Envelope

Every domain endpoint returns a paginated envelope:

```json
{
  "data": [ /* array of domain records */ ],
  "meta": {
    "page": 1,
    "page_size": 100,
    "total_records": 37124,
    "total_pages": 372
  }
}
```

### Example Requests

```bash
# Health check
curl http://localhost:8090/health

# First page of AE records
curl http://localhost:8090/api/v1/ae

# Filter by study and site, 10 records per page
curl "http://localhost:8090/api/v1/ae?study=STUDY-001&site=SITE-001&page_size=10"

# Second page of lab results
curl "http://localhost:8090/api/v1/lb?page=2&page_size=50"
```

## Testing

### Automated Tests
Run the full test suite (105 tests covering all endpoints, pagination, and filters):
```bash
uv run pytest tests/ -v
```

### Manual / HTTP Client Tests
`test_client.http` contains ready-to-run requests for every endpoint and common
scenarios (filters, pagination, edge cases). Open it in VS Code with the
[REST Client](https://marketplace.visualstudio.com/items?itemName=humao.rest-client)
extension and click **Send Request** above any call.

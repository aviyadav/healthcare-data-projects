"""
Smoke test and performance benchmark for the Clinical Data API.

Sections
--------
1. PyIceberg catalog metadata
2. In-memory cache loading (timing)
3. Hot-path query benchmark  (cache warm)
4. FastAPI endpoints via TestClient (lifespan triggers cache load)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── 1. Catalog ──────────────────────────────────────────────────────────────
print("=" * 60)
print("1. PyIceberg catalog")
print("=" * 60)

if os.path.exists("clinical_catalog.db"):
    os.remove("clinical_catalog.db")

from app.config import settings
from app.data.catalog import initialize_catalog, list_domain_metadata

initialize_catalog(settings.catalog_uri, settings.catalog_warehouse)
domains_meta = list_domain_metadata()
for d in domains_meta:
    print(
        f"  {d['code']:4s}  fields={d['field_count']:2d}  partitions={d['partition_fields']}"
    )
assert len(domains_meta) == 6
print("  ✓ catalog OK\n")

# ── 2. Cache loading ─────────────────────────────────────────────────────────
print("=" * 60)
print("2. In-memory cache load (PyArrow → Polars, parallel)")
print("=" * 60)

from app.data.cache import get_cache_stats, load_all_domains

t0 = time.perf_counter()
load_all_domains(settings.data_root)
elapsed = time.perf_counter() - t0

stats = get_cache_stats()
print(f"  Loaded {len(stats['loaded_domains'])} domains in {elapsed * 1000:.0f} ms")
print(f"  Total rows:   {stats['total_rows']:,}")
print(f"  Total memory: {stats['total_memory_mb']:.1f} MB")
print()
for domain, ds in stats["domains"].items():
    print(
        f"  {domain:4s}  rows={ds['rows']:8,}  mem={ds['memory_mb']:6.1f} MB  "
        f"load={ds['load_time_ms']:6.0f} ms"
    )
assert len(stats["loaded_domains"]) == 6
print("  ✓ cache OK\n")

# ── 3. Hot-path benchmark ────────────────────────────────────────────────────
print("=" * 60)
print("3. Hot-path query benchmark (cache warm, page_size=3000)")
print("=" * 60)

from pathlib import Path

from app.data.reader import query_domain

data_root = Path("../clinical_data_output")

bench_cases = [
    (
        "AE",
        {
            "study": "STUDY-001",
            "site": "SITE-001",
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
    (
        "CM",
        {
            "study": "STUDY-001",
            "site": None,
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
    (
        "DM",
        {
            "study": "STUDY-001",
            "site": None,
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
    (
        "LB",
        {
            "study": "STUDY-002",
            "site": None,
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
    (
        "TV",
        {
            "study": "STUDY-001",
            "site": None,
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
    (
        "VS",
        {
            "study": "STUDY-001",
            "site": "SITE-001",
            "subject": None,
            "visit": None,
            "form": None,
        },
    ),
]

import orjson

for domain, filters in bench_cases:
    t0 = time.perf_counter()
    records, total = query_domain(
        domain_path=data_root / domain,
        domain=domain,
        filters=filters,
        page=1,
        page_size=3000,
    )
    query_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    payload = orjson.dumps({"data": records, "meta": {"total": total}})
    serial_ms = (time.perf_counter() - t1) * 1000

    print(
        f"  {domain:4s}  total={total:8,}  page={len(records):4,}  "
        f"query={query_ms:6.1f} ms  serial={serial_ms:5.1f} ms  "
        f"payload={len(payload) // 1024:4d} KB"
    )
    assert isinstance(records, list)
    assert total >= 0
print("  ✓ benchmark OK\n")

# ── 4. FastAPI endpoints ──────────────────────────────────────────────────────
print("=" * 60)
print("4. FastAPI endpoints via TestClient (lifespan = warm cache)")
print("=" * 60)

from app.main import app
from fastapi.testclient import TestClient

# Use TestClient as context manager so the ASGI lifespan runs (loads cache).
with TestClient(app) as client:
    # Health
    r = client.get("/health")
    assert r.status_code == 200, f"/health → {r.status_code}"
    health = r.json()
    print(
        f"  GET /health         → {r.status_code}  status={health['status']}  "
        f"cached_domains={health['cache']['loaded_domains']}"
    )

    # Domains
    r = client.get("/api/v1/domains")
    assert r.status_code == 200
    print(f"  GET /api/v1/domains → {r.status_code}  total={r.json()['total']}")

    # Each domain endpoint — measure response time
    endpoints = [
        ("/api/v1/ae", {"study": "STUDY-001", "site": "SITE-001", "page_size": "3000"}),
        ("/api/v1/cm", {"study": "STUDY-001", "page_size": "3000"}),
        ("/api/v1/dm", {"study": "STUDY-001", "page_size": "3000"}),
        ("/api/v1/lb", {"study": "STUDY-002", "page_size": "3000"}),
        ("/api/v1/tv", {"study": "STUDY-001", "page_size": "3000"}),
        ("/api/v1/vs", {"study": "STUDY-001", "site": "SITE-002", "page_size": "3000"}),
    ]

    for path, params in endpoints:
        t0 = time.perf_counter()
        r = client.get(path, params=params)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"{path} → {r.status_code}\n{r.text[:300]}"
        body = r.json()
        meta = body["meta"]
        print(
            f"  GET {path:15s}  → {r.status_code}  "
            f"total={meta['total_records']:8,}  "
            f"page={len(body['data']):4,}  "
            f"time={elapsed:7.1f} ms"
        )

print("  ✓ endpoints OK\n")

print("=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)

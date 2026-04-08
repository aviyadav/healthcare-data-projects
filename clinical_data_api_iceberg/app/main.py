"""
Clinical Data API – application entry point.

Startup sequence
----------------
1. ``lifespan`` context manager fires.
2. All six SDTM domain DataFrames are loaded from Hive-partitioned Parquet
   into process memory **in parallel** (via ``cache.load_all_domains`` running
   in a ThreadPoolExecutor through ``asyncio.to_thread``).  After this point
   every query is served from RAM – no file I/O on the hot path.
3. PyIceberg SqlCatalog is initialised (SQLite-backed schema registry).
4. FastAPI begins serving requests.

Response serialisation
----------------------
Each router returns ``fastapi.Response(content=orjson.dumps({...}), media_type="application/json")``
directly.  orjson is 5-10× faster than the Python stdlib ``json`` module and
natively serialises ``datetime.date`` / ``datetime.datetime`` objects to
ISO-8601 strings without extra converters.

Returning a pre-serialised ``Response`` bypasses Pydantic's per-record
model-validation step on the output path.  The ``response_model`` annotation
is kept on every route so that FastAPI still generates the correct OpenAPI /
Swagger schema – validation just does not run at response time, which
eliminates the overhead of instantiating thousands of Pydantic models per page
request.

Gunicorn + preload_app
----------------------
When started via Gunicorn with ``preload_app = True`` (see ``gunicorn.conf.py``)
the Python module is imported once in the master process.  Workers are forked
afterwards and each runs its own ASGI lifespan, so ``load_all_domains`` is
called once per worker.  On Linux/macOS the OS copy-on-write mechanism keeps
the Arrow column buffers (the bulk of in-memory data) physically shared across
workers as long as they remain read-only.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .config import settings
from .data import cache
from .data.catalog import initialize_catalog
from .routers import ae, cm, dm, lb, system, tv, vs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler.

    *Startup*
        1. Load all six SDTM domains into memory in parallel.
           ``asyncio.to_thread`` offloads the blocking I/O to a worker thread
           so the event loop stays responsive during the load.
        2. Initialise the PyIceberg SqlCatalog schema registry.

    *Shutdown*
        Nothing to tear down – SQLite handles its own connections and the
        in-memory DataFrames are GC'd with the process.
    """
    logger.info("=== Clinical Data API starting up ===")
    logger.info("Data root  : %s", settings.data_root)
    logger.info("Catalog URI: %s", settings.catalog_uri)

    if not settings.data_root.exists():
        logger.warning(
            "Data root '%s' does not exist – all queries will return empty results.",
            settings.data_root,
        )
    else:
        available = sorted(d.name for d in settings.data_root.iterdir() if d.is_dir())
        logger.info("Data root confirmed – domains on disk: %s", available)

    # ── 1. Warm in-memory cache ──────────────────────────────────────────────
    # Run the blocking PyArrow / Polars I/O in a thread so the event loop is
    # not blocked.  All six domains are loaded in parallel inside the function.
    logger.info("Loading domain DataFrames into memory …")
    try:
        await asyncio.to_thread(cache.load_all_domains, settings.data_root)
        stats = cache.get_cache_stats()
        logger.info(
            "Cache ready – domains=%s  total_rows=%d  total_memory=%.1f MB",
            stats["loaded_domains"],
            stats["total_rows"],
            stats["total_memory_mb"],
        )
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: queries fall back to the slower on-disk reader.
        logger.warning(
            "Cache loading failed – queries will use the slow file-based path. "
            "Reason: %s",
            exc,
            exc_info=settings.debug,
        )

    # ── 2. Initialise PyIceberg catalog ─────────────────────────────────────
    try:
        initialize_catalog(
            uri=settings.catalog_uri,
            warehouse=settings.catalog_warehouse,
        )
        logger.info("PyIceberg catalog initialised.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "PyIceberg catalog init failed – /api/v1/domains will still work "
            "(reads schemas from in-memory PyIceberg objects). Reason: %s",
            exc,
            exc_info=settings.debug,
        )

    logger.info("=== Clinical Data API ready ===")
    yield  # ── application is live here ──────────────────────────────────────
    logger.info("=== Clinical Data API shutting down ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(ae.router)
app.include_router(cm.router)
app.include_router(dm.router)
app.include_router(lb.router)
app.include_router(tv.router)
app.include_router(vs.router)
app.include_router(system.router)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect bare root to the interactive Swagger UI."""
    return RedirectResponse(url="/docs")

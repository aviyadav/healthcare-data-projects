"""
Clinical Data API — FastAPI application factory.

Registers all domain routers under /api/v1 and provides:
- GET /health                 — liveness probe
- GET /api/v1/domains         — metadata listing all available domains
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import ae, cm, dm, lb, tv, vs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Clinical Data API v%s starting — DB: %s | Workers: %s",
        settings.app_version,
        settings.db_path,
        settings.workers,
    )
    yield
    logger.info("Clinical Data API shutting down.")


app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS middleware — permissive for development; tighten in production
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Domain routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(ae.router, prefix=API_PREFIX)
app.include_router(cm.router, prefix=API_PREFIX)
app.include_router(dm.router, prefix=API_PREFIX)
app.include_router(lb.router, prefix=API_PREFIX)
app.include_router(tv.router, prefix=API_PREFIX)
app.include_router(vs.router, prefix=API_PREFIX)

# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------

DOMAIN_METADATA = [
    {"domain": "AE", "name": "Adverse Events",           "endpoint": f"{API_PREFIX}/ae"},
    {"domain": "CM", "name": "Concomitant Medications",  "endpoint": f"{API_PREFIX}/cm"},
    {"domain": "DM", "name": "Demographics",             "endpoint": f"{API_PREFIX}/dm"},
    {"domain": "LB", "name": "Laboratory Results",       "endpoint": f"{API_PREFIX}/lb"},
    {"domain": "TV", "name": "Trial Visits",             "endpoint": f"{API_PREFIX}/tv"},
    {"domain": "VS", "name": "Vital Signs",              "endpoint": f"{API_PREFIX}/vs"},
]


@app.get("/health", tags=["System"], summary="Health check")
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 OK when the service is running."""
    return JSONResponse(content={"status": "ok", "version": settings.app_version})


@app.get(f"{API_PREFIX}/domains", tags=["System"], summary="List available clinical domains")
async def list_domains() -> JSONResponse:
    """Return metadata for all available SDTM clinical domains."""
    return JSONResponse(content={"domains": DOMAIN_METADATA})




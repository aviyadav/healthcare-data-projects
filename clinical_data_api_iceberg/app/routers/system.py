"""System / utility endpoints: health check, cache stats, and domain listing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import orjson
from fastapi import APIRouter, HTTPException, Response

from ..data.cache import get_cache_stats
from ..data.catalog import list_domain_metadata

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System"])


@router.get(
    "/health",
    summary="Health check",
    description="Liveness probe — returns 200 OK when the service is running. Also includes in-memory cache statistics.",
    operation_id="health_check_health_get",
)
def health_check() -> Response:
    cache_stats = get_cache_stats()
    return Response(
        content=orjson.dumps(
            {
                "status": "ok",
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "service": "Clinical Data API",
                "cache": cache_stats,
            }
        ),
        media_type="application/json",
    )


@router.get(
    "/api/v1/domains",
    summary="List available clinical domains",
    description="Return metadata for all available SDTM clinical domains.",
    operation_id="list_domains_api_v1_domains_get",
)
def list_domains() -> Response:
    try:
        domains = list_domain_metadata()
    except Exception as exc:
        logger.error("Failed to retrieve domain metadata: %s", exc)
        raise HTTPException(
            status_code=500, detail="Could not retrieve domain metadata."
        ) from exc
    return Response(
        content=orjson.dumps({"domains": domains, "total": len(domains)}),
        media_type="application/json",
    )

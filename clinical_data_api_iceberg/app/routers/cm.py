"""Concomitant Medications (CM) router."""

from __future__ import annotations

from typing import Optional

import orjson
from fastapi import APIRouter, Query, Response

from ..config import settings
from ..data.reader import query_domain
from ..models import CMRecord, PaginatedResponse

router = APIRouter(tags=["Concomitant Medications (CM)"])


@router.get(
    "/api/v1/cm",
    response_model=PaginatedResponse[CMRecord],
    summary="List Concomitant Medications",
    description=(
        "Retrieve paginated concomitant medication records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
    operation_id="get_concomitant_medications_api_v1_cm_get",
)
def get_concomitant_medications(
    study: Optional[str] = Query(None, description="Filter by STUDY identifier"),
    site: Optional[str] = Query(None, description="Filter by SITE identifier"),
    subject: Optional[str] = Query(None, description="Filter by SUBJECT identifier"),
    visit: Optional[str] = Query(None, description="Filter by VISIT name"),
    form: Optional[str] = Query(None, description="Filter by FORM name"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        1000, ge=1, le=3000, description="Records per page (max 3000)"
    ),
) -> Response:
    records, total = query_domain(
        domain_path=settings.data_root / "CM",
        domain="CM",
        filters={
            "study": study,
            "site": site,
            "subject": subject,
            "visit": visit,
            "form": form,
        },
        page=page,
        page_size=page_size,
    )
    total_pages = max(1, -(-total // page_size)) if total > 0 else 1
    return Response(
        content=orjson.dumps(
            {
                "data": records,
                "meta": {
                    "page": page,
                    "page_size": page_size,
                    "total_records": total,
                    "total_pages": total_pages,
                },
            }
        ),
        media_type="application/json",
    )

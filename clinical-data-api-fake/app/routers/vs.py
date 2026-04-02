"""
GET /api/v1/vs — Vital Signs domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import VSRecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/vs", tags=["Vital Signs (VS)"])

_TABLE = "VS"


@router.get(
    "",
    response_model=PaginatedResponse[VSRecord],
    summary="List Vital Signs",
    description=(
        "Retrieve paginated vital sign records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_vital_signs(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[VSRecord]:
    records, meta = await fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[VSRecord](data=records, meta=meta)

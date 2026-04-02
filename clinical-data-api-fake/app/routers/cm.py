"""
GET /api/v1/cm — Concomitant Medications domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import CMRecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/cm", tags=["Concomitant Medications (CM)"])

_TABLE = "CM"


@router.get(
    "",
    response_model=PaginatedResponse[CMRecord],
    summary="List Concomitant Medications",
    description=(
        "Retrieve paginated concomitant medication records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_concomitant_medications(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[CMRecord]:
    records, meta = fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[CMRecord](data=records, meta=meta)

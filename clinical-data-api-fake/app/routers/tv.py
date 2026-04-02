"""
GET /api/v1/tv — Trial Visits domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import TVRecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/tv", tags=["Trial Visits (TV)"])

_TABLE = "TV"


@router.get(
    "",
    response_model=PaginatedResponse[TVRecord],
    summary="List Trial Visits",
    description=(
        "Retrieve paginated trial visit schedule records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_trial_visits(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[TVRecord]:
    records, meta = await fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[TVRecord](data=records, meta=meta)

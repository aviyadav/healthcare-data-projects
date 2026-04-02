"""
GET /api/v1/ae — Adverse Events domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import AERecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/ae", tags=["Adverse Events (AE)"])

_TABLE = "AE"


@router.get(
    "",
    response_model=PaginatedResponse[AERecord],
    summary="List Adverse Events",
    description=(
        "Retrieve paginated adverse event records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_adverse_events(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[AERecord]:
    records, meta = await fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[AERecord](data=records, meta=meta)

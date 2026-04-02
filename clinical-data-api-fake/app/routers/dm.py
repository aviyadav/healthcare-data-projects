"""
GET /api/v1/dm — Demographics domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import DMRecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/dm", tags=["Demographics (DM)"])

_TABLE = "DM"


@router.get(
    "",
    response_model=PaginatedResponse[DMRecord],
    summary="List Demographics",
    description=(
        "Retrieve paginated demographics records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_demographics(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[DMRecord]:
    records, meta = fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[DMRecord](data=records, meta=meta)

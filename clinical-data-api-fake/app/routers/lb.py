"""
GET /api/v1/lb — Laboratory Results domain router.
"""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.models.schemas import LBRecord, PaginatedResponse
from app.routers.dependencies import common_filters
from app.services.query_service import fetch_domain_data

router = APIRouter(prefix="/lb", tags=["Laboratory Results (LB)"])

_TABLE = "LB"


@router.get(
    "",
    response_model=PaginatedResponse[LBRecord],
    summary="List Laboratory Results",
    description=(
        "Retrieve paginated laboratory result records. "
        "Supports filtering by study, site, subject, visit, and form."
    ),
)
async def get_laboratory_results(
    filters: Annotated[dict, Depends(common_filters)],
) -> PaginatedResponse[LBRecord]:
    records, meta = fetch_domain_data(table=_TABLE, **filters)
    return PaginatedResponse[LBRecord](data=records, meta=meta)

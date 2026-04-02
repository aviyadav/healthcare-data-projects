"""
Shared router utilities — common query parameter definitions used by all domain routers.
"""
from typing import Annotated, Optional

from fastapi import Query

from app.config import get_settings

_settings = get_settings()


def common_filters(
    study: Annotated[Optional[str], Query(description="Filter by STUDY identifier")] = None,
    site: Annotated[Optional[str], Query(description="Filter by SITE identifier")] = None,
    subject: Annotated[Optional[str], Query(description="Filter by SUBJECT identifier")] = None,
    visit: Annotated[Optional[str], Query(description="Filter by VISIT name")] = None,
    form: Annotated[Optional[str], Query(description="Filter by FORM name")] = None,
    page: Annotated[int, Query(ge=1, description="Page number (1-indexed)")] = 1,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=_settings.max_page_size,
            description=f"Records per page (max {_settings.max_page_size})",
        ),
    ] = _settings.default_page_size,
):
    """Dependency that bundles shared filter + pagination query params."""
    return {
        "study": study,
        "site": site,
        "subject": subject,
        "visit": visit,
        "form": form,
        "page": page,
        "page_size": page_size,
    }

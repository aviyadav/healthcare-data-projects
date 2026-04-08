from __future__ import annotations

from datetime import date
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Generic pagination helpers
# ---------------------------------------------------------------------------

T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Metadata included with every paginated response."""

    page: int = Field(..., description="Current page number (1-indexed)")
    page_size: int = Field(..., description="Number of records per page")
    total_records: int = Field(
        ..., description="Total matching records across all pages"
    )
    total_pages: int = Field(..., description="Total number of pages")

    model_config = {
        "json_schema_extra": {
            "example": {
                "page": 1,
                "page_size": 100,
                "total_pages": 50,
                "total_records": 5000,
            }
        }
    }


class PaginatedResponse(BaseModel, Generic[T]):
    """Wrapper returned by every list endpoint."""

    data: List[T] = Field(..., description="List of records for the current page")
    meta: PaginationMeta = Field(..., description="Pagination metadata")


# ---------------------------------------------------------------------------
# AE – Adverse Events
# ---------------------------------------------------------------------------


class AERecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # AE-specific fields
    AESEQ: Optional[int] = None
    AETERM: Optional[str] = None
    AEDECOD: Optional[str] = None
    AEBODSYS: Optional[str] = None
    AESTDTC: Optional[date] = None
    AEENDTC: Optional[date] = None
    AESEV: Optional[str] = None
    AEREL: Optional[str] = None
    AEOUT: Optional[str] = None
    AE_INCIDENT_GROUP: Optional[str] = None

    # Partition / identifier columns
    SITEID: Optional[str] = None
    STUDYID: Optional[str] = None
    USUBJID: Optional[str] = None


# ---------------------------------------------------------------------------
# CM – Concomitant Medications
# ---------------------------------------------------------------------------


class CMRecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # CM-specific fields
    CMSEQ: Optional[int] = None
    CMTRT: Optional[str] = None
    CMDECOD: Optional[str] = None
    CMCAT: Optional[str] = None
    CMSTDTC: Optional[date] = None
    CMENDTC: Optional[date] = None
    CMDOSE: Optional[float] = None
    CMDOSU: Optional[str] = None
    CMDOSFRM: Optional[str] = None
    CMROUTE: Optional[str] = None
    CMDOSFRQ: Optional[str] = None

    # Partition / identifier columns
    SITEID: Optional[str] = None
    STUDYID: Optional[str] = None
    USUBJID: Optional[str] = None


# ---------------------------------------------------------------------------
# DM – Demographics
# ---------------------------------------------------------------------------


class DMRecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # DM-specific fields
    AGE: Optional[int] = None
    SEX: Optional[str] = None
    RACE: Optional[str] = None
    COUNTRY: Optional[str] = None
    DMDTC: Optional[date] = None
    ARM: Optional[str] = None

    # Partition / identifier columns
    SITEID: Optional[str] = None
    STUDYID: Optional[str] = None
    USUBJID: Optional[str] = None


# ---------------------------------------------------------------------------
# LB – Laboratory Results
# ---------------------------------------------------------------------------


class LBRecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # LB-specific fields
    LBTESTCD: Optional[str] = None
    LBTEST: Optional[str] = None
    LBORRES: Optional[float] = None
    LBORRESU: Optional[str] = None
    LBSTNRLO: Optional[float] = None
    LBSTNRHI: Optional[float] = None
    LBDTC: Optional[date] = None

    # Partition / identifier columns
    SITEID: Optional[str] = None
    STUDYID: Optional[str] = None
    USUBJID: Optional[str] = None


# ---------------------------------------------------------------------------
# TV – Trial Visits
# ---------------------------------------------------------------------------


class TVRecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # TV-specific fields
    VISITNUM: Optional[int] = None
    TVSTRL: Optional[int] = None
    TVENRL: Optional[int] = None
    ARMCD: Optional[str] = None

    # Partition / identifier column (TV has no SITEID/USUBJID)
    STUDYID: Optional[str] = None


# ---------------------------------------------------------------------------
# VS – Vital Signs
# ---------------------------------------------------------------------------


class VSRecord(BaseModel):
    STUDY: Optional[str] = None
    SITE: Optional[str] = None
    SUBJECT: Optional[str] = None
    VISIT: Optional[str] = None
    FORM: Optional[str] = None
    DOMAIN: Optional[str] = None

    # VS-specific fields
    VSTESTCD: Optional[str] = None
    VSTEST: Optional[str] = None
    VSORRES: Optional[float] = None
    VSORRESU: Optional[str] = None
    VSDTC: Optional[date] = None

    # Partition / identifier columns
    SITEID: Optional[str] = None
    STUDYID: Optional[str] = None
    USUBJID: Optional[str] = None

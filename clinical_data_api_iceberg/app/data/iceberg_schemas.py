"""
PyIceberg schema definitions for all six CDISC SDTM clinical domains.

Each schema mirrors the OpenAPI response models and the actual Parquet column
types observed in the data:
  - large_string  → StringType()
  - int64         → LongType()
  - double        → DoubleType()
  - date32[day]   → DateType()
  - dictionary<…> → StringType()   (partition columns, dict-encoded in files)

Partition specs describe the Hive-style directory layout used on disk so that
the catalog can reason about data layout without scanning every file.
"""

from __future__ import annotations

from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    DateType,
    DoubleType,
    LongType,
    NestedField,
    StringType,
)

# ---------------------------------------------------------------------------
# Helper – build a NestedField with required=False (all columns are nullable)
# ---------------------------------------------------------------------------


def _f(field_id: int, name: str, field_type, doc: str = "") -> NestedField:
    return NestedField(
        field_id=field_id,
        name=name,
        field_type=field_type,
        required=False,
        doc=doc,
    )


# ---------------------------------------------------------------------------
# AE – Adverse Events
# Partition layout: STUDYID / SITEID / USUBJID / AE_INCIDENT_GROUP
# ---------------------------------------------------------------------------

AE_SCHEMA = Schema(
    # Common SDTM columns (present in every domain)
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # AE-specific columns
    _f(7, "AESEQ", LongType(), "Sequence number of the adverse event"),
    _f(8, "AETERM", StringType(), "Reported adverse event term"),
    _f(9, "AEDECOD", StringType(), "Dictionary-derived term (MedDRA PT)"),
    _f(10, "AEBODSYS", StringType(), "Body system / SOC"),
    _f(11, "AESTDTC", DateType(), "Start date of adverse event"),
    _f(12, "AEENDTC", DateType(), "End date of adverse event"),
    _f(13, "AESEV", StringType(), "Severity / intensity"),
    _f(14, "AEREL", StringType(), "Causality (relationship to study drug)"),
    _f(15, "AEOUT", StringType(), "Outcome of adverse event"),
    # Partition / identifier columns (dictionary-encoded in files)
    _f(16, "AE_INCIDENT_GROUP", StringType(), "Adverse event incident group"),
    _f(17, "SITEID", StringType(), "Site ID (partition key)"),
    _f(18, "STUDYID", StringType(), "Study ID (partition key)"),
    _f(19, "USUBJID", StringType(), "Unique subject ID (partition key)"),
)

AE_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=18, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
    PartitionField(
        source_id=17, field_id=1001, transform=IdentityTransform(), name="SITEID"
    ),
    PartitionField(
        source_id=19, field_id=1002, transform=IdentityTransform(), name="USUBJID"
    ),
    PartitionField(
        source_id=16,
        field_id=1003,
        transform=IdentityTransform(),
        name="AE_INCIDENT_GROUP",
    ),
)


# ---------------------------------------------------------------------------
# CM – Concomitant Medications
# Partition layout: STUDYID / SITEID / USUBJID
# ---------------------------------------------------------------------------

CM_SCHEMA = Schema(
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # CM-specific columns
    _f(7, "CMSEQ", LongType(), "Sequence number of concomitant medication"),
    _f(8, "CMTRT", StringType(), "Reported name of drug / medication"),
    _f(9, "CMDECOD", StringType(), "Standardized medication name (WHODrug)"),
    _f(10, "CMCAT", StringType(), "Category for medication"),
    _f(11, "CMSTDTC", DateType(), "Start date of medication"),
    _f(12, "CMENDTC", DateType(), "End date of medication"),
    _f(13, "CMDOSE", DoubleType(), "Dose per administration"),
    _f(14, "CMDOSU", StringType(), "Units of dose"),
    _f(15, "CMDOSFRM", StringType(), "Dose form (tablet, capsule, …)"),
    _f(16, "CMROUTE", StringType(), "Route of administration"),
    _f(17, "CMDOSFRQ", StringType(), "Dosing frequency per interval"),
    # Partition / identifier columns
    _f(18, "SITEID", StringType(), "Site ID (partition key)"),
    _f(19, "STUDYID", StringType(), "Study ID (partition key)"),
    _f(20, "USUBJID", StringType(), "Unique subject ID (partition key)"),
)

CM_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=19, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
    PartitionField(
        source_id=18, field_id=1001, transform=IdentityTransform(), name="SITEID"
    ),
    PartitionField(
        source_id=20, field_id=1002, transform=IdentityTransform(), name="USUBJID"
    ),
)


# ---------------------------------------------------------------------------
# DM – Demographics
# Partition layout: STUDYID / SITEID / USUBJID
# ---------------------------------------------------------------------------

DM_SCHEMA = Schema(
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # DM-specific columns
    _f(7, "AGE", LongType(), "Age of subject at time of consent"),
    _f(8, "SEX", StringType(), "Sex of subject"),
    _f(9, "RACE", StringType(), "Race of subject"),
    _f(10, "COUNTRY", StringType(), "Country of participation"),
    _f(11, "DMDTC", DateType(), "Date of demographics collection"),
    _f(12, "ARM", StringType(), "Description of planned arm"),
    # Partition / identifier columns
    _f(13, "SITEID", StringType(), "Site ID (partition key)"),
    _f(14, "STUDYID", StringType(), "Study ID (partition key)"),
    _f(15, "USUBJID", StringType(), "Unique subject ID (partition key)"),
)

DM_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=14, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
    PartitionField(
        source_id=13, field_id=1001, transform=IdentityTransform(), name="SITEID"
    ),
    PartitionField(
        source_id=15, field_id=1002, transform=IdentityTransform(), name="USUBJID"
    ),
)


# ---------------------------------------------------------------------------
# LB – Laboratory Results
# Partition layout: STUDYID / SITEID / USUBJID
# ---------------------------------------------------------------------------

LB_SCHEMA = Schema(
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # LB-specific columns
    _f(7, "LBTESTCD", StringType(), "Lab test short name"),
    _f(8, "LBTEST", StringType(), "Lab test long name"),
    _f(9, "LBORRES", DoubleType(), "Result or finding in original units"),
    _f(10, "LBORRESU", StringType(), "Original units"),
    _f(11, "LBSTNRLO", DoubleType(), "Reference range lower limit (standard units)"),
    _f(12, "LBSTNRHI", DoubleType(), "Reference range upper limit (standard units)"),
    _f(13, "LBDTC", DateType(), "Date of lab specimen collection"),
    # Partition / identifier columns
    _f(14, "SITEID", StringType(), "Site ID (partition key)"),
    _f(15, "STUDYID", StringType(), "Study ID (partition key)"),
    _f(16, "USUBJID", StringType(), "Unique subject ID (partition key)"),
)

LB_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=15, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
    PartitionField(
        source_id=14, field_id=1001, transform=IdentityTransform(), name="SITEID"
    ),
    PartitionField(
        source_id=16, field_id=1002, transform=IdentityTransform(), name="USUBJID"
    ),
)


# ---------------------------------------------------------------------------
# TV – Trial Visits
# Partition layout: STUDYID  (no SITEID / USUBJID in TV files)
# ---------------------------------------------------------------------------

TV_SCHEMA = Schema(
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # TV-specific columns
    _f(7, "VISITNUM", LongType(), "Visit number"),
    _f(8, "TVSTRL", LongType(), "Planned study day of start of visit window"),
    _f(9, "TVENRL", LongType(), "Planned study day of end of visit window"),
    _f(10, "ARMCD", StringType(), "Planned arm code"),
    # Partition / identifier column (TV only has STUDYID partition)
    _f(11, "STUDYID", StringType(), "Study ID (partition key)"),
)

TV_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=11, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
)


# ---------------------------------------------------------------------------
# VS – Vital Signs
# Partition layout: STUDYID / SITEID / USUBJID
# ---------------------------------------------------------------------------

VS_SCHEMA = Schema(
    _f(1, "STUDY", StringType(), "Study identifier (human-readable)"),
    _f(2, "SITE", StringType(), "Site identifier (human-readable)"),
    _f(3, "SUBJECT", StringType(), "Subject identifier (human-readable)"),
    _f(4, "VISIT", StringType(), "Visit name"),
    _f(5, "FORM", StringType(), "Form / CRF page name"),
    _f(6, "DOMAIN", StringType(), "SDTM domain code"),
    # VS-specific columns
    _f(7, "VSTESTCD", StringType(), "Vital signs test short name"),
    _f(8, "VSTEST", StringType(), "Vital signs test long name"),
    _f(9, "VSORRES", DoubleType(), "Result or finding in original units"),
    _f(10, "VSORRESU", StringType(), "Original units"),
    _f(11, "VSDTC", DateType(), "Date of vital signs measurement"),
    # Partition / identifier columns
    _f(12, "SITEID", StringType(), "Site ID (partition key)"),
    _f(13, "STUDYID", StringType(), "Study ID (partition key)"),
    _f(14, "USUBJID", StringType(), "Unique subject ID (partition key)"),
)

VS_PARTITION_SPEC = PartitionSpec(
    PartitionField(
        source_id=13, field_id=1000, transform=IdentityTransform(), name="STUDYID"
    ),
    PartitionField(
        source_id=12, field_id=1001, transform=IdentityTransform(), name="SITEID"
    ),
    PartitionField(
        source_id=14, field_id=1002, transform=IdentityTransform(), name="USUBJID"
    ),
)


# ---------------------------------------------------------------------------
# Registry – convenient lookup by domain code
# ---------------------------------------------------------------------------

DOMAIN_SCHEMAS: dict[str, Schema] = {
    "AE": AE_SCHEMA,
    "CM": CM_SCHEMA,
    "DM": DM_SCHEMA,
    "LB": LB_SCHEMA,
    "TV": TV_SCHEMA,
    "VS": VS_SCHEMA,
}

DOMAIN_PARTITION_SPECS: dict[str, PartitionSpec] = {
    "AE": AE_PARTITION_SPEC,
    "CM": CM_PARTITION_SPEC,
    "DM": DM_PARTITION_SPEC,
    "LB": LB_PARTITION_SPEC,
    "TV": TV_PARTITION_SPEC,
    "VS": VS_PARTITION_SPEC,
}

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "AE": "Adverse Events – records of any undesirable medical occurrences.",
    "CM": "Concomitant Medications – drugs taken alongside the study treatment.",
    "DM": "Demographics – baseline characteristics of each study subject.",
    "LB": "Laboratory Results – clinical lab test values and reference ranges.",
    "TV": "Trial Visits – planned visit schedule and window definitions.",
    "VS": "Vital Signs – blood pressure, heart rate, temperature, weight, etc.",
}

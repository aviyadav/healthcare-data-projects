"""
PyIceberg catalog bootstrap.

Responsibilities
----------------
* Create (or reuse) a SQLite-backed SqlCatalog at startup.
* Register a "clinical" namespace.
* For every SDTM domain (AE, CM, DM, LB, TV, VS) attempt to create an Iceberg
  table with the correct schema and partition spec if it does not already exist.
* Expose ``list_domain_metadata()`` which builds rich schema/partition metadata
  by reading *directly* from the in-memory PyIceberg Schema objects defined in
  ``iceberg_schemas.py``.  This makes the metadata endpoint resilient to
  warehouse write failures (e.g. Windows drive-letter URI issues) while still
  leveraging PyIceberg's type system fully.

Design note – why read schemas in-memory?
------------------------------------------
PyIceberg's ``SqlCatalog`` writes Iceberg table-metadata JSON files into a
warehouse directory.  On Windows the path ``C:/...`` is parsed by
``urllib.parse.urlparse`` as having scheme ``c`` (the drive letter), which
PyIceberg/PyArrow cannot map to a filesystem implementation.  Rather than
fighting that platform quirk, ``list_domain_metadata()`` constructs its
response from the canonical ``DOMAIN_SCHEMAS`` / ``DOMAIN_PARTITION_SPECS``
dicts, which are pure-Python PyIceberg objects and carry no I/O.  The
SqlCatalog registration is still *attempted* at startup so the catalog DB
reflects the domain tables on platforms where the warehouse path is writable;
failures are logged as warnings and do not affect API availability.
"""

from __future__ import annotations

import logging
from typing import Any

from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import (
    NamespaceAlreadyExistsError,
    TableAlreadyExistsError,
)

from .iceberg_schemas import (
    DOMAIN_DESCRIPTIONS,
    DOMAIN_PARTITION_SPECS,
    DOMAIN_SCHEMAS,
)

logger = logging.getLogger(__name__)

# Module-level singleton – initialised once during app lifespan startup.
_catalog: SqlCatalog | None = None

# The single Iceberg namespace that owns all domain tables.
_NAMESPACE = "clinical"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_namespace(catalog: SqlCatalog) -> None:
    """Create the 'clinical' namespace if it does not already exist."""
    try:
        catalog.create_namespace(_NAMESPACE)
        logger.info("Created Iceberg namespace '%s'.", _NAMESPACE)
    except NamespaceAlreadyExistsError:
        logger.debug("Iceberg namespace '%s' already exists – reusing.", _NAMESPACE)


def _ensure_tables(catalog: SqlCatalog) -> None:
    """
    Register each domain as an Iceberg table inside the catalog.

    Failures (e.g. warehouse path issues on Windows) are logged as warnings
    and silently swallowed so the API still starts successfully.
    """
    for domain_code, schema in DOMAIN_SCHEMAS.items():
        identifier = f"{_NAMESPACE}.{domain_code}"
        partition_spec = DOMAIN_PARTITION_SPECS[domain_code]

        try:
            catalog.create_table(
                identifier=identifier,
                schema=schema,
                partition_spec=partition_spec,
                properties={
                    "description": DOMAIN_DESCRIPTIONS.get(domain_code, ""),
                    "domain_code": domain_code,
                    "write.format.default": "parquet",
                },
            )
            logger.info("Registered Iceberg table '%s'.", identifier)

        except TableAlreadyExistsError:
            logger.debug(
                "Iceberg table '%s' already registered – skipping.", identifier
            )

        except Exception as exc:  # noqa: BLE001
            # Non-fatal: log and continue so the API still starts.
            logger.warning(
                "Could not register Iceberg table '%s': %s",
                identifier,
                exc,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_catalog(uri: str, warehouse: str) -> SqlCatalog:
    """
    Initialise the module-level SqlCatalog singleton.

    Parameters
    ----------
    uri:
        SQLAlchemy connection string, e.g. ``sqlite:///clinical_catalog.db``.
    warehouse:
        Path hint for the Iceberg warehouse root.  Used by PyIceberg when
        persisting table-metadata JSON files.  May fail silently on Windows
        due to drive-letter URI parsing; domain queries are unaffected.

    Returns
    -------
    SqlCatalog
        The initialised catalog instance (also stored as ``_catalog``).
    """
    global _catalog  # noqa: PLW0603

    logger.info(
        "Initialising PyIceberg SqlCatalog  uri=%s  warehouse=%s", uri, warehouse
    )

    _catalog = SqlCatalog(
        _NAMESPACE,
        **{
            "uri": uri,
            "warehouse": warehouse,
        },
    )

    _ensure_namespace(_catalog)
    _ensure_tables(_catalog)

    logger.info(
        "PyIceberg catalog ready – %d domain schemas registered.",
        len(DOMAIN_SCHEMAS),
    )
    return _catalog


def get_catalog() -> SqlCatalog:
    """
    Return the module-level catalog singleton.

    Raises
    ------
    RuntimeError
        If :func:`initialize_catalog` has not been called yet.
    """
    if _catalog is None:
        raise RuntimeError(
            "PyIceberg catalog has not been initialised. "
            "Call initialize_catalog() during application startup."
        )
    return _catalog


# ---------------------------------------------------------------------------
# Metadata helpers used by the /api/v1/domains endpoint
# ---------------------------------------------------------------------------


def list_domain_metadata() -> list[dict[str, Any]]:
    """
    Return rich schema and partition metadata for all SDTM domains.

    **Implementation note** – this function reads *directly* from the
    in-memory PyIceberg ``Schema`` and ``PartitionSpec`` objects defined in
    ``iceberg_schemas.py``.  It does not require the SqlCatalog warehouse to
    be writable (avoiding Windows drive-letter URI issues) and is therefore
    always available regardless of catalog persistence state.

    Each returned dict contains:

    ``code``
        Domain code (e.g. ``"AE"``).
    ``description``
        Plain-English description of the domain.
    ``field_count``
        Number of fields in the Iceberg schema.
    ``fields``
        List of ``{id, name, type, required, doc}`` dicts – one per field.
    ``partition_fields``
        Ordered list of partition field *names* matching the Hive directory
        structure on disk (e.g. ``["STUDYID", "SITEID", "USUBJID"]``).
    ``iceberg_schema_id``
        Iceberg schema version identifier from the ``Schema`` object.
    ``record_count``
        Always ``-1``; a live count requires a full data scan (use the
        domain endpoint with no filters to get the real total).
    """
    results: list[dict[str, Any]] = []

    for domain_code, schema in DOMAIN_SCHEMAS.items():
        spec = DOMAIN_PARTITION_SPECS[domain_code]

        # Build field list directly from PyIceberg NestedField objects.
        fields: list[dict[str, Any]] = [
            {
                "id": field.field_id,
                "name": field.name,
                "type": str(field.field_type),
                "required": field.required,
                "doc": field.doc or "",
            }
            for field in schema.fields
        ]

        # Partition field names in layout order.
        partition_field_names: list[str] = [pf.name for pf in spec.fields]

        results.append(
            {
                "code": domain_code,
                "description": DOMAIN_DESCRIPTIONS.get(domain_code, ""),
                "field_count": len(fields),
                "fields": fields,
                "partition_fields": partition_field_names,
                "iceberg_schema_id": schema.schema_id,
                "record_count": -1,  # live count requires a data scan
            }
        )

    return results

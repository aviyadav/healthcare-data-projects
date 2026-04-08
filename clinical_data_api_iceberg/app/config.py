from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLINICAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_title: str = "Clinical Data API"
    app_description: str = (
        "A high-performance async REST API for accessing clinical trial data "
        "across the AE, CM, DM, LB, TV, and VS CDISC SDTM domains. "
        "Backed by PyIceberg catalog with Polars/PyArrow for zero-copy columnar processing."
    )
    app_version: str = "1.0.0"

    # Root directory that contains AE/, CM/, DM/, LB/, TV/, VS/ sub-folders
    data_root: Path = Path(__file__).resolve().parent.parent / "clinical_data_output"

    # SQLite URI for the PyIceberg SQL catalog (relative path = next to this package)
    catalog_uri: str = "sqlite:///" + str(
        Path(__file__).resolve().parent.parent / "clinical_catalog.db"
    ).replace("\\", "/")

    # Dedicated Iceberg warehouse directory (separate from the raw Parquet data).
    # PyIceberg writes table metadata (JSON) here; the actual Parquet files stay
    # in data_root.  We use a plain forward-slash path so PyArrow's LocalFileSystem
    # handles it correctly on Windows (file:// URIs with drive letters cause issues).
    @property
    def catalog_warehouse(self) -> str:
        warehouse = Path(__file__).resolve().parent.parent / "iceberg_warehouse"
        warehouse.mkdir(parents=True, exist_ok=True)
        return str(warehouse).replace("\\", "/")

    debug: bool = False


settings = Settings()

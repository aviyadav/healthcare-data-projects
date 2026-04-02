"""
Application configuration using pydantic-settings.
Environment variables override defaults.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    db_path: str = str(Path(__file__).parent.parent / "clinical_data.duckdb")

    # Pagination
    default_page_size: int = 1000
    max_page_size: int = 3000

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Single worker by default — DuckDB file locking makes multiple workers unsafe
    workers: int = 1

    # App metadata
    app_title: str = "Clinical Data API"
    app_description: str = (
        "A high-performance async REST API for accessing clinical trial data "
        "across the AE, CM, DM, LB, TV, and VS CDISC SDTM domains. "
        "Backed by DuckDB with Polars/PyArrow for zero-copy columnar processing."
    )
    app_version: str = "1.0.0"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance (created once per process)."""
    return Settings()

"""
Shared pytest fixtures for the clinical data API test suite.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Session-scoped synchronous TestClient.
    Using scope='session' means the FastAPI app is instantiated once for all tests,
    which also means DuckDB connections are reused (faster test runs).
    """
    with TestClient(app) as c:
        yield c

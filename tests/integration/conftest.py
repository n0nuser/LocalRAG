from __future__ import annotations

import os

import httpx
import pytest

BASE_URL = os.getenv("LOCALRAG_TEST_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session", autouse=True)
def require_api(base_url: str) -> None:
    try:
        response = httpx.get(f"{base_url}/health", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError:
        pytest.fail("LocalRAG API is not reachable; the Compose integration job must start it")


@pytest.fixture(scope="session")
def api_key() -> str:
    value = os.getenv("LOCALRAG_TEST_API_KEY", "")
    if not value:
        pytest.fail("LOCALRAG_TEST_API_KEY is required for authenticated integration tests")
    return value


@pytest.fixture(scope="session")
def auth_enabled(base_url: str) -> bool:
    """Probe a cheap protected endpoint and infer whether API key auth is enabled."""
    try:
        response = httpx.get(f"{base_url}/collections", timeout=10.0)
    except httpx.HTTPError:
        pytest.fail("Could not probe API authentication; integration setup is incomplete")
    assert response.status_code == 401, "Compose integration must enable API authentication"
    return True

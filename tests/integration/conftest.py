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
        pytest.skip("LocalRAG API not reachable - run docker compose up first")


@pytest.fixture(scope="session")
def api_key() -> str:
    return os.getenv("LOCALRAG_TEST_API_KEY", "")


@pytest.fixture(scope="session")
def auth_enabled(base_url: str) -> bool:
    """Probe a cheap protected endpoint and infer whether API key auth is enabled."""
    try:
        response = httpx.get(f"{base_url}/collections", timeout=10.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 401

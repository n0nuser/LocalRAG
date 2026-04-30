from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def test_health(base_url: str) -> None:
    response = httpx.get(f"{base_url}/health", timeout=10.0)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "ollama_ok" in body
    assert "chroma_path" in body
    assert "collections" in body


def test_metrics_endpoint(base_url: str) -> None:
    response = httpx.get(f"{base_url}/metrics", timeout=10.0)
    assert response.status_code == 200
    assert "localrag_query_duration_seconds" in response.text


def test_api_key_missing_returns_401(base_url: str, auth_enabled: bool) -> None:
    if not auth_enabled:
        pytest.skip("Auth is not enabled in this environment.")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "hello"},
        timeout=30.0,
    )
    assert response.status_code == 401


def test_api_key_invalid_returns_401(base_url: str, auth_enabled: bool) -> None:
    if not auth_enabled:
        pytest.skip("Auth is not enabled in this environment.")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "hello"},
        headers={"X-API-Key": "invalid-key"},
        timeout=30.0,
    )
    assert response.status_code == 401


def test_api_key_valid_passthrough(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if not auth_enabled:
        pytest.skip("Auth is not enabled in this environment.")
    if not api_key:
        pytest.skip("Set LOCALRAG_TEST_API_KEY to run authenticated integration tests.")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "hello"},
        headers=_headers(api_key),
        timeout=30.0,
    )
    assert response.status_code != 401


def test_ingest_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.skip("Set LOCALRAG_TEST_API_KEY to run authenticated integration tests.")
    response = httpx.post(
        f"{base_url}/ingest",
        json={"path": "/app/docs/architecture.md"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks_added"] >= 1


def test_query_json_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.skip("Set LOCALRAG_TEST_API_KEY to run authenticated integration tests.")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "What is LocalRAG?"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code in {200, 502}


def test_query_stream_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.skip("Set LOCALRAG_TEST_API_KEY to run authenticated integration tests.")
    response = httpx.post(
        f"{base_url}/query/stream",
        json={"question": "Give me one sentence about LocalRAG."},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code in {200, 502}
    if response.status_code == 200:
        assert any(line.startswith("data:") for line in response.text.splitlines())


def test_agent_query_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.skip("Set LOCALRAG_TEST_API_KEY to run authenticated integration tests.")
    response = httpx.post(
        f"{base_url}/agent/query",
        json={"question": "What is LocalRAG?"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code != 500

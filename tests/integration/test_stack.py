from __future__ import annotations

import httpx
import pytest

pytestmark = pytest.mark.integration


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def test_health_and_readiness(base_url: str) -> None:
    response = httpx.get(f"{base_url}/health", timeout=10.0)
    assert response.status_code == 200
    assert response.json().get("status") == "ok"
    assert "chroma_path" not in response.json()

    readiness = httpx.get(f"{base_url}/ready", timeout=10.0)
    assert readiness.status_code == 200
    assert readiness.json() == {"status": "ok"}


def test_metrics_endpoint(base_url: str) -> None:
    response = httpx.get(f"{base_url}/metrics", timeout=10.0)
    assert response.status_code == 200
    assert "localrag_query_duration_seconds" in response.text


def test_api_key_missing_returns_401(base_url: str, auth_enabled: bool) -> None:
    if not auth_enabled:
        pytest.fail("Auth must be enabled for the integration stack")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "hello"},
        timeout=30.0,
    )
    assert response.status_code == 401


def test_api_key_invalid_returns_401(base_url: str, auth_enabled: bool) -> None:
    if not auth_enabled:
        pytest.fail("Auth must be enabled for the integration stack")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "hello"},
        headers={"X-API-Key": "invalid-key"},
        timeout=30.0,
    )
    assert response.status_code == 401


def test_api_key_valid_passthrough(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if not auth_enabled:
        pytest.fail("Auth must be enabled for the integration stack")
    if not api_key:
        pytest.fail("LOCALRAG_TEST_API_KEY is required")
    response = httpx.get(f"{base_url}/collections", headers=_headers(api_key), timeout=10.0)
    assert response.status_code == 200


def test_ingest_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.fail("LOCALRAG_TEST_API_KEY is required")
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
        pytest.fail("LOCALRAG_TEST_API_KEY is required")
    response = httpx.post(
        f"{base_url}/query",
        json={"question": "What is LocalRAG?", "model": "llama3.2:latest"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code == 200


def test_query_stream_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.fail("LOCALRAG_TEST_API_KEY is required")
    response = httpx.post(
        f"{base_url}/query/stream",
        json={"question": "Give me one sentence about LocalRAG.", "model": "llama3.2:latest"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code == 200
    assert any(line.startswith(("data:", "event:")) for line in response.text.splitlines())


def test_agent_query_endpoint(base_url: str, auth_enabled: bool, api_key: str) -> None:
    if auth_enabled and not api_key:
        pytest.fail("LOCALRAG_TEST_API_KEY is required")
    response = httpx.post(
        f"{base_url}/agent/query",
        json={"question": "What is LocalRAG?"},
        headers=_headers(api_key),
        timeout=60.0,
    )
    assert response.status_code != 500


def test_collection_lifecycle(base_url: str, api_key: str) -> None:
    headers = _headers(api_key)
    ingest = httpx.post(
        f"{base_url}/ingest",
        json={"path": "/app/docs/architecture.md"},
        headers=headers,
        timeout=60.0,
    )
    assert ingest.status_code == 200

    listed = httpx.get(f"{base_url}/collections", headers=headers, timeout=10.0)
    assert listed.status_code == 200
    collection_name = listed.json()["collections"][0]
    deleted = httpx.delete(
        f"{base_url}/collections/{collection_name}", headers=headers, timeout=10.0
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "ok"}

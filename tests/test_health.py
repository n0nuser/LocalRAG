from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from localrag.api.main import app
from localrag.application.container import Container
from localrag.settings import Settings

ApiContainer = Callable[..., Container]


class HealthyRepository:
    def list_collection_names(self) -> list[str]:
        return ["localrag"]


def test_health_is_liveness_only_and_does_not_call_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no container to reach, any dependency lookup would fail the request.
    monkeypatch.delattr(app.state, "container", raising=False)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@respx.mock
def test_ready_returns_503_without_required_dependencies(api_container: ApiContainer) -> None:
    api_container(
        Settings(ollama_base_url="http://ollama:11434"), collection_repository=HealthyRepository()
    )
    respx.get("http://ollama:11434/api/tags").mock(return_value=httpx.Response(503))

    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


@respx.mock
def test_ready_does_not_expose_storage_details(api_container: ApiContainer) -> None:
    settings = Settings(ollama_base_url="http://ollama:11434", chroma_persist_path="/secret/path")
    api_container(settings, collection_repository=HealthyRepository())
    respx.get("http://ollama:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )

    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "/secret/path" not in response.text
    assert "localrag" not in response.text

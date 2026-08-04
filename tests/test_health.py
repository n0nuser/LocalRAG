from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from localrag.api.dependencies import get_api_settings, get_collection_repository
from localrag.api.main import app
from localrag.settings import Settings


class HealthyRepository:
    def list_collection_names(self) -> list[str]:
        return ["localrag"]


def test_health_is_liveness_only_and_does_not_call_dependencies() -> None:
    app.dependency_overrides[get_collection_repository] = lambda: (_ for _ in ()).throw(
        AssertionError("liveness must not check dependencies")
    )
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    app.dependency_overrides.clear()


@respx.mock
def test_ready_returns_503_without_required_dependencies() -> None:
    app.dependency_overrides[get_api_settings] = lambda: Settings(
        ollama_base_url="http://ollama:11434"
    )
    app.dependency_overrides[get_collection_repository] = HealthyRepository
    respx.get("http://ollama:11434/api/tags").mock(return_value=httpx.Response(503))

    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
    app.dependency_overrides.clear()


@respx.mock
def test_ready_does_not_expose_storage_details() -> None:
    settings = Settings(ollama_base_url="http://ollama:11434", chroma_persist_path="/secret/path")
    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_collection_repository] = HealthyRepository
    respx.get("http://ollama:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )

    response = TestClient(app).get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "/secret/path" not in response.text
    assert "localrag" not in response.text
    app.dependency_overrides.clear()

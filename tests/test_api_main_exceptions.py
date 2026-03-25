from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
import respx
from fastapi.testclient import TestClient

from localrag.api.dependencies import (
    get_api_settings,
    get_collection_repository,
    get_engine,
)
from localrag.api.main import app
from localrag.rag.exceptions import RetrievalError
from localrag.settings import Settings


@respx.mock
def test_validation_error_is_mapped_to_422() -> None:
    client = TestClient(app)

    respx.get("http://ollama:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m"}]}),
    )

    # Missing required `question` field.
    response = client.post("/query", json={})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert any("question" in str(err.get("loc")) for err in body["detail"])


@respx.mock
def test_unhandled_exception_results_in_500() -> None:
    base_url = "http://ollama:11434"
    settings = Settings(ollama_base_url=base_url, chroma_persist_path="./data/chroma")

    class ExplodingRepo:
        def list_collection_names(self) -> list[str]:
            raise RuntimeError("boom")

    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_collection_repository] = lambda: ExplodingRepo()

    client = TestClient(app, raise_server_exceptions=False)
    respx.get(f"{base_url}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m"}]}),
    )

    response = client.get("/health")
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"

    app.dependency_overrides.clear()


@dataclass
class FailingRetriever:
    def retrieve(self, question: str, n_results: int | None = None) -> list[object]:
        raise RetrievalError(HTTPStatus.BAD_GATEWAY, "Embedding service unavailable.")


@dataclass
class FailingQueryEngine:
    retriever: FailingRetriever = field(default_factory=FailingRetriever)


def test_query_maps_retrieval_failure_to_502() -> None:
    app.dependency_overrides[get_engine] = lambda: FailingQueryEngine()
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/query", json={"question": "Hi"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Embedding service unavailable."

    app.dependency_overrides.clear()

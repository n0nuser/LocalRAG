from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus

import httpx
import respx
from fastapi.testclient import TestClient

from localrag.api.main import app
from localrag.application.container import Container
from localrag.rag.exceptions import RetrievalError
from localrag.settings import Settings, get_settings

ApiContainer = Callable[..., Container]


@respx.mock
def test_validation_error_is_mapped_to_422(api_container: ApiContainer) -> None:
    api_container(engine=object())  # the request fails validation before the engine is used
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
def test_unhandled_exception_results_in_500(api_container: ApiContainer) -> None:
    base_url = "http://ollama:11434"
    settings = Settings(ollama_base_url=base_url, chroma_persist_path="./data/chroma")

    class ExplodingRepo:
        def list_collection_names(self) -> list[str]:
            raise RuntimeError("boom")

    api_container(settings, collection_repository=ExplodingRepo())

    client = TestClient(app, raise_server_exceptions=False)
    respx.get(f"{base_url}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m"}]}),
    )

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


@dataclass
class FailingRetriever:
    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[object]:
        _ = (question, n_results, metadata_filter)
        raise RetrievalError(HTTPStatus.BAD_GATEWAY, "Embedding service unavailable.")


@dataclass
class FailingQueryEngine:
    retriever: FailingRetriever = field(default_factory=FailingRetriever)
    settings: Settings = field(default_factory=get_settings)


def test_query_maps_retrieval_failure_to_502(api_container: ApiContainer) -> None:
    api_container(engine=FailingQueryEngine())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/query", json={"question": "Hi"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Embedding service unavailable."

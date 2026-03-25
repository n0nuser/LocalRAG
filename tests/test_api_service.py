from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import respx
from fastapi.testclient import TestClient

from localrag.api.dependencies import (
    get_api_settings,
    get_collection_repository,
    get_ingestion_service,
)
from localrag.api.main import app
from localrag.ingestion.service import IngestionResult
from localrag.settings import Settings


@dataclass
class StubCollectionRepo:
    names: list[str]
    deleted: list[str]

    def list_collection_names(self) -> list[str]:
        return self.names

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)


@dataclass
class StubIngestionService:
    called: list[tuple[Path, bool | None]]
    result: IngestionResult

    def ingest_directory(self, path: Path, recursive: bool | None = None) -> IngestionResult:
        self.called.append((path, recursive))
        return self.result


@respx.mock
def test_health_success_and_collections_endpoints(tmp_path: Path) -> None:
    base_url = "http://ollama:11434"
    settings = Settings(ollama_base_url=base_url, chroma_persist_path=str(tmp_path))
    repo = StubCollectionRepo(names=["col-1", "col-2"], deleted=[])

    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_collection_repository] = lambda: repo
    client = TestClient(app)

    respx.get(f"{base_url}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m"}]}),
    )

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ollama_ok"] is True
    assert health.json()["collections"] == ["col-1", "col-2"]

    collections = client.get("/collections")
    assert collections.status_code == 200
    assert collections.json() == {"collections": ["col-1", "col-2"]}

    deleted = client.delete("/collections/col-1")
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "ok"}
    assert repo.deleted == ["col-1"]

    app.dependency_overrides.clear()


@respx.mock
def test_health_marks_ollama_unreachable_on_http_error() -> None:
    base_url = "http://ollama:11434"
    settings = Settings(ollama_base_url=base_url, chroma_persist_path="./data/chroma")
    repo = StubCollectionRepo(names=["col"], deleted=[])

    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_collection_repository] = lambda: repo
    client = TestClient(app)

    respx.get(f"{base_url}/api/tags").mock(return_value=httpx.Response(500, json={"x": 1}))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ollama_ok"] is False

    app.dependency_overrides.clear()


@respx.mock
def test_health_marks_ollama_unreachable_on_invalid_response() -> None:
    base_url = "http://ollama:11434"
    settings = Settings(ollama_base_url=base_url, chroma_persist_path="./data/chroma")
    repo = StubCollectionRepo(names=["col"], deleted=[])

    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_collection_repository] = lambda: repo
    client = TestClient(app)

    respx.get(f"{base_url}/api/tags").mock(return_value=httpx.Response(200, json={"wrong": True}))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["ollama_ok"] is False

    app.dependency_overrides.clear()


def test_ingest_directory_success_and_forbidden_outside_roots(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    inside_dir = allowed_root / "docs"
    inside_dir.mkdir()

    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_dir = outside_root / "docs"
    outside_dir.mkdir()

    ingestion = StubIngestionService(
        called=[],
        result=IngestionResult(files_processed=2, total_chunks=3, processed_sources=[]),
    )

    settings = Settings(ingest_roots=[str(allowed_root)])
    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion

    client = TestClient(app)
    ok = client.post(
        "/ingest/directory",
        json={"path": str(inside_dir), "recursive": False},
    )
    assert ok.status_code == 200
    assert ok.json() == {"status": "ok", "files_processed": 2, "total_chunks": 3}
    assert ingestion.called[0][0] == inside_dir.resolve()
    assert ingestion.called[0][1] is False

    forbidden = client.post(
        "/ingest/directory",
        json={"path": str(outside_dir), "recursive": False},
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"] == "Path is not under configured ingest roots."

    app.dependency_overrides.clear()

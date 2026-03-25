from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from localrag.api.dependencies import get_api_settings, get_engine, get_ingestion_service
from localrag.api.main import app
from localrag.ingestion.service import IngestionResult
from localrag.settings import Settings


@dataclass
class StubRetriever:
    def retrieve(self, question: str, n_results: int | None = None) -> list[dict[str, Any]]:
        _ = (question, n_results)
        return []


@dataclass
class StubEngine:
    retriever: StubRetriever = field(default_factory=StubRetriever)

    def stream_chat_from_contexts(
        self,
        *,
        contexts: list[dict[str, Any]],
        question: str,
        model: str | None,
    ) -> object:
        _ = (contexts, question, model)
        yield {"type": "token", "token": "hello "}
        yield {"type": "token", "token": "world"}
        yield {"type": "final", "sources": [{"source": "doc.md", "chunk_index": 1}]}


def test_query_streams_events() -> None:
    app.dependency_overrides[get_engine] = lambda: StubEngine()
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hi"})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "hello" in response.text
    assert "event: final" in response.text

    app.dependency_overrides.clear()


@dataclass
class UnusedIngestionService:
    def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
        raise AssertionError(path)

    def ingest_directory(
        self, path: Path, recursive: bool | None = None, embed_model: str | None = None
    ) -> IngestionResult:
        raise AssertionError(path)


def test_ingest_rejects_missing_file() -> None:
    app.dependency_overrides[get_ingestion_service] = lambda: UnusedIngestionService()
    client = TestClient(app)
    missing = Path(__file__).resolve().parent / f"missing_{uuid4()}.txt"
    response = client.post("/ingest", json={"path": str(missing)})
    assert response.status_code == 400
    assert response.json()["detail"] == "Path must be an existing file."
    app.dependency_overrides.clear()


def test_ingest_accepts_percent_encoded_spaces(tmp_path: Path) -> None:
    doc = tmp_path / "my doc.txt"
    doc.write_text("hello", encoding="utf-8")

    @dataclass
    class RecordingIngestionService:
        seen: list[Path]

        def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
            self.seen.append(path)
            return IngestionResult(files_processed=1, total_chunks=1, processed_sources=[str(path)])

        def ingest_directory(
            self, path: Path, recursive: bool | None = None, embed_model: str | None = None
        ) -> IngestionResult:
            raise AssertionError(path)

    recording = RecordingIngestionService(seen=[])
    app.dependency_overrides[get_ingestion_service] = lambda: recording
    client = TestClient(app)
    response = client.post("/ingest", json={"path": str(tmp_path / "my%20doc.txt")})
    assert response.status_code == 200
    assert len(recording.seen) == 1
    assert " " in str(recording.seen[0])
    assert "%20" not in str(recording.seen[0])
    app.dependency_overrides.clear()


def test_ingest_forbidden_outside_roots(tmp_path: Path) -> None:
    inner = tmp_path / "allowed"
    inner.mkdir()
    outer_file = tmp_path / "outside.txt"
    outer_file.write_text("x", encoding="utf-8")
    app.dependency_overrides[get_api_settings] = lambda: Settings(ingest_roots=[str(inner)])
    app.dependency_overrides[get_ingestion_service] = lambda: UnusedIngestionService()
    client = TestClient(app)
    response = client.post("/ingest", json={"path": str(outer_file)})
    assert response.status_code == 403
    app.dependency_overrides.clear()


def test_ingest_directory_rejects_file(tmp_path: Path) -> None:
    file_only = tmp_path / "a.txt"
    file_only.write_text("x", encoding="utf-8")
    app.dependency_overrides[get_ingestion_service] = lambda: UnusedIngestionService()
    client = TestClient(app)
    response = client.post("/ingest/directory", json={"path": str(file_only)})
    assert response.status_code == 400
    assert response.json()["detail"] == "Path must be an existing directory."
    app.dependency_overrides.clear()

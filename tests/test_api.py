from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from localrag.api.main import app
from localrag.application.container import Container
from localrag.ingestion.service import IngestionResult
from localrag.settings import Settings
from localrag.storage.persist_lock import ConcurrentIngestError

ApiContainer = Callable[..., Container]

_STUB_CONTEXTS = [{"source": "doc.md", "chunk_index": 1, "text": "chunk"}]


@dataclass
class StubRetriever:
    def retrieve(
        self,
        question: str,
        n_results: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        _ = (question, n_results, metadata_filter)
        return _STUB_CONTEXTS


@dataclass
class StubEngine:
    retriever: StubRetriever = field(default_factory=StubRetriever)
    settings: Settings = field(default_factory=lambda: Settings(ollama_llm_model="stub-model"))

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
        yield {
            "type": "final",
            "sources": [{"source": "doc.md", "chunk_index": 1}],
            "trace": {
                "mode": "fallback",
                "provider": "ollama",
                "model": "stub-model",
                "latency_ms": 1.0,
                "status": "fallback",
                "fallback_reason": "timeout",
            },
        }

    @staticmethod
    def extract_sources(contexts: list[dict[str, Any]]) -> list[dict[str, object]]:
        return [
            {"source": str(c.get("source", "")), "chunk_index": int(c.get("chunk_index", -1))}
            for c in contexts
        ]


def test_query_json_returns_answer(api_container: ApiContainer) -> None:
    api_container(engine=StubEngine())
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "hello world"
    assert body["sources"][0]["source"] == "doc.md"
    assert "latency_ms" in body
    assert body["model"] == "stub-model"
    assert body["trace"]["status"] == "fallback"


def test_query_json_selects_request_collection(api_container: ApiContainer) -> None:
    selected: list[str] = []

    @dataclass
    class CollectionEngine(StubEngine):
        def for_collection(self, collection: str) -> CollectionEngine:
            selected.append(collection)
            return self

    api_container(engine=CollectionEngine())
    response = TestClient(app).post("/query", json={"question": "Hi", "collection": "experiments"})

    assert response.status_code == 200
    assert selected == ["experiments"]


def test_benchmark_contexts_return_text_and_stable_id(api_container: ApiContainer) -> None:
    class BenchmarkRetriever(StubRetriever):
        def retrieve(self, **_kwargs: object) -> list[dict[str, Any]]:
            return [
                {
                    "source": "/private/doc.md",
                    "chunk_index": 2,
                    "chunk_id": "stable-chunk-id",
                    "text": "actual retrieved passage",
                    "metadata": {"chunk_id": "stable-chunk-id"},
                }
            ]

    @dataclass
    class BenchmarkEngine(StubEngine):
        retriever: BenchmarkRetriever = field(default_factory=BenchmarkRetriever)

    api_container(engine=BenchmarkEngine())
    response = TestClient(app).post("/query/contexts", json={"question": "Hi"})

    assert response.status_code == 200
    assert response.json()["contexts"] == [
        {
            "chunk_id": "stable-chunk-id",
            "text": "actual retrieved passage",
            "source": "/private/doc.md",
            "chunk_index": 2,
        }
    ]


def test_query_streams_events(api_container: ApiContainer) -> None:
    api_container(engine=StubEngine())
    client = TestClient(app)

    response = client.post("/query/stream", json={"question": "Hi"})

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "hello" in response.text
    assert "event: final" in response.text


def test_metrics_endpoint_returns_prometheus_text(api_container: ApiContainer) -> None:
    api_container()
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "python_info" in response.text or "HELP" in response.text


def test_build_info_is_protected(
    monkeypatch: pytest.MonkeyPatch, api_container: ApiContainer
) -> None:
    monkeypatch.setenv("LOCALRAG_BUILD_SHA", "test-sha")
    api_container(Settings(api_key="secret"))
    client = TestClient(app)

    assert client.get("/build-info").status_code == 401
    response = client.get("/build-info", headers={"X-API-Key": "secret"})

    assert response.json() == {"build_sha": "test-sha"}


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({}, 401),
        ({"X-API-Key": "wrong"}, 401),
        ({"X-API-Key": "secret"}, 200),
    ],
)
def test_api_key_enforcement(
    headers: dict[str, str], expected_status: int, api_container: ApiContainer
) -> None:
    api_container(Settings(api_key="secret"), engine=StubEngine())
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hi"}, headers=headers)
    assert response.status_code == expected_status


def test_api_key_disabled_when_not_configured(api_container: ApiContainer) -> None:
    """When API_KEY is empty, all requests pass through without a key."""
    api_container(Settings(api_key=""), engine=StubEngine())
    client = TestClient(app)

    response = client.post("/query", json={"question": "Hi"})
    assert response.status_code == 200


@dataclass
class UnusedIngestionService:
    def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
        raise AssertionError(path)

    def ingest_directory(
        self, path: Path, recursive: bool | None = None, embed_model: str | None = None
    ) -> IngestionResult:
        raise AssertionError(path)


def test_ingest_rejects_missing_file(api_container: ApiContainer) -> None:
    api_container(ingestion_service=UnusedIngestionService())
    client = TestClient(app)
    missing = Path(__file__).resolve().parent / f"missing_{uuid4()}.txt"
    response = client.post("/ingest", json={"path": str(missing)})
    assert response.status_code == 400
    assert response.json()["detail"] == "Path must be an existing file."


def test_ingest_accepts_percent_encoded_spaces(tmp_path: Path, api_container: ApiContainer) -> None:
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
    api_container(ingestion_service=recording)
    client = TestClient(app)
    response = client.post("/ingest", json={"path": str(tmp_path / "my%20doc.txt")})
    assert response.status_code == 200
    assert len(recording.seen) == 1
    assert " " in str(recording.seen[0])
    assert "%20" not in str(recording.seen[0])


def test_ingest_forbidden_outside_roots(tmp_path: Path, api_container: ApiContainer) -> None:
    inner = tmp_path / "allowed"
    inner.mkdir()
    outer_file = tmp_path / "outside.txt"
    outer_file.write_text("x", encoding="utf-8")
    api_container(Settings(ingest_roots=[str(inner)]), ingestion_service=UnusedIngestionService())
    client = TestClient(app)
    response = client.post("/ingest", json={"path": str(outer_file)})
    assert response.status_code == 403


def test_ingest_directory_rejects_file(tmp_path: Path, api_container: ApiContainer) -> None:
    file_only = tmp_path / "a.txt"
    file_only.write_text("x", encoding="utf-8")
    api_container(ingestion_service=UnusedIngestionService())
    client = TestClient(app)
    response = client.post("/ingest/directory", json={"path": str(file_only)})
    assert response.status_code == 400
    assert response.json()["detail"] == "Path must be an existing directory."


def test_ingest_upload_saves_and_ingests_file(tmp_path: Path, api_container: ApiContainer) -> None:
    @dataclass
    class RecordingIngestionService:
        seen: list[Path]

        def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
            self.seen.append(path)
            assert path.read_bytes() == b"hello world"
            return IngestionResult(files_processed=1, total_chunks=3, processed_sources=[str(path)])

    recording = RecordingIngestionService(seen=[])
    api_container(Settings(upload_dir=str(tmp_path)), ingestion_service=recording)
    client = TestClient(app)

    response = client.post(
        "/ingest/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_added"] == 3
    assert len(recording.seen) == 1
    assert recording.seen[0].parent == tmp_path


def test_ingest_upload_rejects_unsupported_extension(
    tmp_path: Path, api_container: ApiContainer
) -> None:
    api_container(Settings(upload_dir=str(tmp_path)), ingestion_service=UnusedIngestionService())
    client = TestClient(app)

    response = client.post(
        "/ingest/upload",
        files={"file": ("payload.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert list(tmp_path.iterdir()) == []


def test_ingest_upload_rejects_oversized_file(tmp_path: Path, api_container: ApiContainer) -> None:
    api_container(
        Settings(upload_dir=str(tmp_path), upload_max_bytes=5),
        ingestion_service=UnusedIngestionService(),
    )
    client = TestClient(app)

    response = client.post(
        "/ingest/upload",
        files={"file": ("big.txt", b"this is more than five bytes", "text/plain")},
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_ingest_upload_is_temporary_by_default(tmp_path: Path, api_container: ApiContainer) -> None:
    @dataclass
    class Recording:
        def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
            return IngestionResult(files_processed=1, total_chunks=1, processed_sources=[str(path)])

    recording = Recording()
    api_container(Settings(upload_dir=str(tmp_path)), ingestion_service=recording)
    response = TestClient(app).post(
        "/ingest/upload", files={"file": ("notes.txt", b"same", "text/plain")}
    )
    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_ingest_upload_retention_quota_removes_oldest(
    tmp_path: Path, api_container: ApiContainer
) -> None:
    @dataclass
    class Recording:
        def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
            return IngestionResult(files_processed=1, total_chunks=1, processed_sources=[str(path)])

    old = tmp_path / "old.txt"
    old.write_bytes(b"old")
    old.touch()
    settings = Settings(
        upload_dir=str(tmp_path), upload_retention_seconds=3600, upload_quota_bytes=3
    )
    api_container(settings, ingestion_service=Recording())
    response = TestClient(app).post(
        "/ingest/upload", files={"file": ("notes.txt", b"new", "text/plain")}
    )
    assert response.status_code == 200
    assert old.exists() is False
    assert len(list(tmp_path.iterdir())) == 1


def test_ingest_returns_409_when_another_process_holds_the_persist_lock(
    tmp_path: Path, api_container: ApiContainer
) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("hello", encoding="utf-8")

    @dataclass
    class ContendedIngestionService:
        def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
            _ = embed_model
            raise ConcurrentIngestError(str(path))

    api_container(ingestion_service=ContendedIngestionService())
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/ingest", json={"path": str(doc)})

    assert response.status_code == HTTPStatus.CONFLICT
    assert str(doc) in response.json()["detail"]

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from localrag.ingestion.service import IngestionResult, IngestionService
from localrag.settings import Settings


@dataclass
class StubEmbedder:
    seen_texts_batches: list[tuple[int, list[str], int, str | None]]

    def embed_texts(
        self, texts: list[str], batch_size: int, *, model: str | None = None
    ) -> list[list[float]]:
        self.seen_texts_batches.append((len(self.seen_texts_batches), texts, batch_size, model))
        return [[float(index)] for index, _ in enumerate(texts)]


@dataclass
class StubVectorStore:
    deleted_sources: list[str]
    added: list[dict[str, object]]
    distinct_sources: list[str] | None = None

    def list_distinct_sources(self) -> list[str]:
        return list(self.distinct_sources or [])

    def delete_by_source(self, source: str) -> None:
        self.deleted_sources.append(source)

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
    ) -> None:
        self.added.append(
            {
                "source": source,
                "chunks": chunks,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )


def test_ingestion_service_ingest_paths_skips_not_allowed_and_empty_chunks(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    disallowed_root = tmp_path / "disallowed"
    disallowed_root.mkdir()

    # Should be ingested (enough text for at least one chunk).
    allowed_file = allowed_root / "a.md"
    allowed_file.write_text("hello world", encoding="utf-8")

    # Allowed but yields no chunks.
    empty_file = allowed_root / "empty.txt"
    empty_file.write_text("   ", encoding="utf-8")

    # Not allowed by ingest_roots.
    disallowed_file = disallowed_root / "b.md"
    disallowed_file.write_text("nope", encoding="utf-8")

    settings = Settings(
        ingest_roots=[str(allowed_root)],
        chunk_chars=5,
        chunk_overlap_chars=0,
        embedding_batch_size=2,
    )
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(deleted_sources=[], added=[], distinct_sources=None)
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    result: IngestionResult = service.ingest_paths([allowed_file, disallowed_file, empty_file])

    assert result.files_processed == 1
    assert result.total_chunks > 0
    assert result.processed_sources == [str(allowed_file.resolve())]

    # delete_by_source + add_chunks should only run for the successful ingest.
    assert vector_store.deleted_sources == [str(allowed_file.resolve())]
    assert len(vector_store.added) == 1

    added = vector_store.added[0]
    assert added["source"] == str(allowed_file.resolve())

    chunks = added["chunks"]
    assert isinstance(chunks, list)
    assert chunks

    metadatas = added["metadatas"]
    assert isinstance(metadatas, list)
    assert all(md.get("file_type") == ".md" for md in metadatas)  # type: ignore[union-attr]

    # Embeddings batching happens inside the embedder; we only need to ensure it ran.
    assert len(embedder.seen_texts_batches) == 1
    _, seen_texts, seen_batch_size, seen_model = embedder.seen_texts_batches[0]
    assert seen_batch_size == settings.embedding_batch_size
    assert seen_texts == chunks
    assert seen_model is None


def test_ingestion_service_ingest_file_delegates_to_ingest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    path = allowed_root / "a.txt"
    path.write_text("hi", encoding="utf-8")

    settings = Settings(ingest_roots=[str(allowed_root)])
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(deleted_sources=[], added=[], distinct_sources=None)
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    expected = IngestionResult(files_processed=1, total_chunks=0, processed_sources=[str(path)])

    called: list[list[Path]] = []

    def fake_ingest_paths(paths: list[Path], **kwargs: object) -> IngestionResult:
        _ = kwargs
        called.append(paths)
        return expected

    # Avoid type complexity: this is just a test seam.
    monkeypatch.setattr(service, "ingest_paths", fake_ingest_paths)  # type: ignore[arg-type]

    out = service.ingest_file(path)
    assert out == expected
    assert called == [[path]]


def test_ingestion_service_ingest_directory_uses_settings_ingest_recursive_when_recursive_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()

    settings = Settings(ingest_roots=[str(root)], ingest_recursive=False)
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(deleted_sources=[], added=[], distinct_sources=None)
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    captured: list[bool] = []

    def fake_list_supported_files(_path: Path, recursive: bool) -> list[Path]:
        captured.append(recursive)
        return []

    expected = IngestionResult(files_processed=0, total_chunks=0, processed_sources=[])

    monkeypatch.setattr(
        "localrag.ingestion.service.list_supported_files",  # type: ignore[arg-type]
        fake_list_supported_files,
    )
    monkeypatch.setattr(
        service,
        "ingest_paths",
        lambda _paths, **_kw: expected,  # type: ignore[arg-type]
    )

    service.ingest_directory(root, recursive=None)
    assert captured == [False]


def test_ingestion_service_ingest_paths_re_raises_parse_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    path = allowed_root / "a.md"
    path.write_text("x", encoding="utf-8")

    settings = Settings(ingest_roots=[str(allowed_root)])
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(deleted_sources=[], added=[], distinct_sources=None)
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    def fake_parse_file(_path: Path) -> str:
        raise ValueError("parse failed")

    monkeypatch.setattr("localrag.ingestion.service.parse_file", fake_parse_file)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="parse failed"):
        service.ingest_paths([path])


def test_ingestion_service_ingest_paths_passes_embed_model_to_embedder(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    path = allowed_root / "a.md"
    path.write_text("hello world wide", encoding="utf-8")

    settings = Settings(ingest_roots=[str(allowed_root)], chunk_chars=100, chunk_overlap_chars=0)
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(deleted_sources=[], added=[], distinct_sources=None)
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    service.ingest_paths([path], embed_model="custom-embed")

    assert embedder.seen_texts_batches[0][3] == "custom-embed"


def test_ingestion_service_rebuild_reingests_distinct_sources(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    kept = allowed_root / "kept.md"
    kept.write_text("hello world", encoding="utf-8")
    missing = str(allowed_root / "gone.md")

    settings = Settings(ingest_roots=[str(allowed_root)], chunk_chars=100, chunk_overlap_chars=0)
    embedder = StubEmbedder(seen_texts_batches=[])
    vector_store = StubVectorStore(
        deleted_sources=[],
        added=[],
        distinct_sources=[str(kept.resolve()), missing],
    )
    service = IngestionService(settings=settings, embedder=embedder, vector_store=vector_store)

    result = service.rebuild_collection()

    assert missing in result.missing_sources
    assert vector_store.deleted_sources[0] == missing
    assert str(kept.resolve()) in result.processed_sources
    assert result.files_processed == 1

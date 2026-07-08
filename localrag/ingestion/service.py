from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from localrag.ingestion.chunker import chunk_text
from localrag.ingestion.embedder import OllamaEmbedder
from localrag.ingestion.loader import list_supported_files, parse_file
from localrag.ingestion.structural_chunker import Chunk, chunk_document
from localrag.settings import Settings, is_path_allowed
from localrag.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FailedSource:
    """A source that still failed after the one-time end-of-batch retry."""

    source: str
    error: str


@dataclass
class IngestionResult:
    files_processed: int
    total_chunks: int
    processed_sources: list[str]
    failed_sources: list[FailedSource] = field(default_factory=list)


@dataclass
class RebuildCollectionResult:
    """Outcome of re-embedding every chunk for sources currently in the vector store."""

    files_processed: int
    total_chunks: int
    processed_sources: list[str]
    missing_sources: list[str]
    failed_sources: list[FailedSource] = field(default_factory=list)
    skipped_unchanged_sources: list[str] = field(default_factory=list)


@dataclass
class IngestionService:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore
    bm25_index: Any | None = None

    def ingest_file(self, path: Path, embed_model: str | None = None) -> IngestionResult:
        return self.ingest_paths([path], embed_model=embed_model)

    def ingest_directory(
        self, path: Path, recursive: bool | None = None, embed_model: str | None = None
    ) -> IngestionResult:
        should_recurse = self.settings.ingest_recursive if recursive is None else recursive
        files = list_supported_files(path, recursive=should_recurse)
        logger.info(
            "ingest_directory_discovered path=%s recursive=%s file_count=%s",
            path.resolve(),
            should_recurse,
            len(files),
        )
        return self.ingest_paths(files, embed_model=embed_model)

    def rebuild_collection(self, embed_model: str | None = None) -> RebuildCollectionResult:
        sources = self.vector_store.list_distinct_sources()
        stored_hashes = self._stored_content_hashes()
        missing_sources: list[str] = []
        skipped_unchanged: list[str] = []
        paths_to_ingest: list[Path] = []
        for source in sources:
            path = Path(source)
            if not path.is_file():
                missing_sources.append(source)
                self.vector_store.delete_by_source(source)
                logger.warning("rebuild_skip_missing_file source=%s", source)
                continue
            current_hash = _file_content_hash(path)
            if stored_hashes.get(source) == current_hash:
                skipped_unchanged.append(source)
                continue
            paths_to_ingest.append(path)
        ingest = self.ingest_paths(paths_to_ingest, embed_model=embed_model)
        logger.info("rebuild_skipped_unchanged count=%s", len(skipped_unchanged))
        return RebuildCollectionResult(
            files_processed=ingest.files_processed,
            total_chunks=ingest.total_chunks,
            processed_sources=ingest.processed_sources,
            missing_sources=sorted(missing_sources),
            failed_sources=ingest.failed_sources,
            skipped_unchanged_sources=sorted(skipped_unchanged),
        )

    def _stored_content_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for _chunk_id, _document, metadata in self.vector_store.get_all_chunks():
            source = metadata.get("source")
            content_hash = metadata.get("content_hash")
            if isinstance(source, str) and isinstance(content_hash, str) and source not in hashes:
                hashes[source] = content_hash
        return hashes

    def ingest_paths(self, paths: list[Path], embed_model: str | None = None) -> IngestionResult:
        total_chunks = 0
        files_processed = 0
        processed_sources: list[str] = []
        retry_queue: list[Path] = []

        for resolved_path in self._allowed_paths(paths):
            try:
                chunks_added = self._ingest_one(resolved_path, embed_model)
            except Exception as exc:  # retried once below, not fatal yet
                logger.warning("ingest_file_failed_will_retry path=%s error=%s", resolved_path, exc)
                retry_queue.append(resolved_path)
                continue
            if chunks_added is None:
                continue
            files_processed += 1
            total_chunks += chunks_added
            processed_sources.append(str(resolved_path))

        failed_sources: list[FailedSource] = []
        if retry_queue:
            logger.info("ingest_retry_start count=%s", len(retry_queue))
        for resolved_path in retry_queue:
            try:
                chunks_added = self._ingest_one(resolved_path, embed_model)
            except Exception as exc:  # collected for the caller, not raised
                logger.error("ingest_file_failed_permanently path=%s error=%s", resolved_path, exc)
                failed_sources.append(FailedSource(source=str(resolved_path), error=str(exc)))
                continue
            if chunks_added is None:
                continue
            files_processed += 1
            total_chunks += chunks_added
            processed_sources.append(str(resolved_path))

        # Rebuilding the BM25 corpus is O(total chunks); do it once per batch, not per file.
        if self.bm25_index is not None and files_processed > 0:
            self.bm25_index.refresh()

        return IngestionResult(
            files_processed=files_processed,
            total_chunks=total_chunks,
            processed_sources=processed_sources,
            failed_sources=failed_sources,
        )

    def _allowed_paths(self, paths: list[Path]) -> list[Path]:
        allowed: list[Path] = []
        for path in paths:
            resolved_path = path.resolve()
            if not is_path_allowed(resolved_path, self.settings.ingest_roots):
                logger.warning(
                    "ingest_skipped_not_allowed path=%s roots=%s",
                    resolved_path,
                    self.settings.ingest_roots,
                )
                continue
            allowed.append(resolved_path)
        return allowed

    def _ingest_one(self, resolved_path: Path, embed_model: str | None) -> int | None:
        """Parse, chunk, embed, and upsert one file. Returns chunks added, or None if skipped."""
        logger.debug("ingest_parse_start path=%s", resolved_path)
        text = parse_file(resolved_path)
        structural_chunks = self._build_chunks(text=text, file_type=resolved_path.suffix.lower())
        chunks = [chunk.text for chunk in structural_chunks]
        if not chunks:
            logger.warning("ingest_skipped_no_chunks path=%s", resolved_path)
            return None

        source = str(resolved_path)
        logger.debug("ingest_embed_start path=%s chunk_count=%s", resolved_path, len(chunks))

        embeddings = self.embedder.embed_texts(
            chunks,
            self.settings.embedding_batch_size,
            model=embed_model,
        )
        # Only drop the old vectors once the new embeddings are in hand, so a failed
        # embed call leaves the previous (still valid) vectors for this source in place.
        self.vector_store.delete_by_source(source)
        created_at = datetime.now(UTC).isoformat()
        content_hash = _file_content_hash(resolved_path)
        source_mtime = resolved_path.stat().st_mtime
        git_commit = _git_commit_for_path(resolved_path) or ""
        metadatas = [
            {
                "source": source,
                "file_type": resolved_path.suffix.lower(),
                "chunk_index": index,
                "heading_path": chunk.heading_path,
                "chunk_type": chunk.chunk_type,
                "ingested_at": created_at,
                "content_hash": content_hash,
                "source_mtime": source_mtime,
                "git_commit": git_commit,
                "tenant_id": self.settings.tenant_id,
            }
            for index, chunk in enumerate(structural_chunks)
        ]
        self.vector_store.add_chunks(
            source=source,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("ingest_file_success path=%s chunks=%s", resolved_path, len(chunks))
        return len(chunks)

    def _build_chunks(self, text: str, file_type: str) -> list[Chunk]:
        if self.settings.chunking_mode == "fixed":
            fixed_chunks = chunk_text(
                text=text,
                chunk_chars=self.settings.chunk_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            return [
                Chunk(text=chunk, heading_path="", chunk_type="fixed") for chunk in fixed_chunks
            ]
        return chunk_document(text=text, file_type=file_type, settings=self.settings)


def _file_content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit_for_path(path: Path) -> str | None:
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "log", "-1", "--format=%H", "--", path.name],  # noqa: S607
            cwd=path.parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = result.stdout.strip()
    return commit or None

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hermit.ingestion.chunker import chunk_text
from hermit.ingestion.embedder import OllamaEmbedder
from hermit.ingestion.loader import list_supported_files, parse_file
from hermit.settings import Settings, is_path_allowed
from hermit.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    files_processed: int
    total_chunks: int
    processed_sources: list[str]


@dataclass
class IngestionService:
    settings: Settings
    embedder: OllamaEmbedder
    vector_store: VectorStore

    def ingest_file(self, path: Path) -> IngestionResult:
        return self.ingest_paths([path])

    def ingest_directory(self, path: Path, recursive: bool | None = None) -> IngestionResult:
        should_recurse = self.settings.ingest_recursive if recursive is None else recursive
        files = list_supported_files(path, recursive=should_recurse)
        logger.info(
            "ingest_directory_discovered path=%s recursive=%s file_count=%s",
            path.resolve(),
            should_recurse,
            len(files),
        )
        return self.ingest_paths(files)

    def ingest_paths(self, paths: list[Path]) -> IngestionResult:
        total_chunks = 0
        files_processed = 0
        processed_sources: list[str] = []

        for path in paths:
            resolved_path = path.resolve()
            if not is_path_allowed(resolved_path, self.settings.ingest_roots):
                logger.warning(
                    "ingest_skipped_not_allowed path=%s roots=%s",
                    resolved_path,
                    self.settings.ingest_roots,
                )
                continue

            logger.debug("ingest_parse_start path=%s", resolved_path)
            try:
                text = parse_file(resolved_path)
            except Exception:
                logger.exception("ingest_parse_failed path=%s", resolved_path)
                raise
            chunks = chunk_text(
                text=text,
                chunk_chars=self.settings.chunk_chars,
                overlap_chars=self.settings.chunk_overlap_chars,
            )
            if not chunks:
                logger.warning("ingest_skipped_no_chunks path=%s", resolved_path)
                continue

            source = str(resolved_path)
            self.vector_store.delete_by_source(source)
            logger.debug("ingest_embed_start path=%s chunk_count=%s", resolved_path, len(chunks))

            embeddings = self.embedder.embed_texts(chunks, self.settings.embedding_batch_size)
            created_at = datetime.now(UTC).isoformat()
            metadatas = [
                {
                    "source": source,
                    "file_type": resolved_path.suffix.lower(),
                    "chunk_index": index,
                    "ingested_at": created_at,
                }
                for index, _ in enumerate(chunks)
            ]
            self.vector_store.add_chunks(
                source=source,
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            files_processed += 1
            total_chunks += len(chunks)
            processed_sources.append(source)
            logger.info(
                "ingest_file_success path=%s chunks=%s",
                resolved_path,
                len(chunks),
            )

        return IngestionResult(
            files_processed=files_processed,
            total_chunks=total_chunks,
            processed_sources=processed_sources,
        )

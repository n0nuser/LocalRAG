from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QueryRequest:
    question: str
    model: str | None = None
    n_results: int | None = None
    metadata_filter: dict[str, str] | None = None
    collection: str | None = None


@dataclass(frozen=True)
class IngestFileRequest:
    path: str
    embed_model: str | None = None


@dataclass(frozen=True)
class IngestDirectoryRequest:
    path: str
    recursive: bool | None = None
    embed_model: str | None = None


@dataclass(frozen=True)
class RebuildCollectionRequest:
    embed_model: str | None = None


@dataclass(frozen=True)
class SourceRef:
    source: str
    chunk_index: int
    heading_path: str | None = None
    chunk_type: str | None = None


@dataclass(frozen=True)
class FailedSourceRef:
    source: str
    error: str


@dataclass(frozen=True)
class QueryResponse:
    answer: str
    sources: list[SourceRef]
    latency_ms: float
    model: str
    provider: str
    low_confidence: bool
    trace: dict[str, object] | None = None


@dataclass(frozen=True)
class QueryContextsResponse:
    contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class IngestFileResponse:
    status: str
    chunks_added: int
    source: str


@dataclass(frozen=True)
class IngestDirectoryResponse:
    status: str
    files_processed: int
    total_chunks: int
    failed_sources: list[FailedSourceRef] = field(default_factory=list)


@dataclass(frozen=True)
class IngestJobResponse:
    job_id: str
    status: str


@dataclass(frozen=True)
class IngestJobStatusResponse:
    job_id: str
    status: str
    result: dict[str, object] | None
    error: str | None


@dataclass(frozen=True)
class CollectionListResponse:
    collections: list[str]


@dataclass(frozen=True)
class CollectionDeleteResponse:
    status: str


@dataclass(frozen=True)
class RebuildCollectionResponse:
    status: str
    files_processed: int
    total_chunks: int
    missing_sources: list[str]
    failed_sources: list[FailedSourceRef]
    skipped_unchanged_sources: list[str]


@dataclass(frozen=True)
class ReadinessResponse:
    status: str

"""The chunk record: the one owner of chunk identity, stored metadata, and retrieval contexts.

Every other module writes and reads chunk metadata through these types instead
of by string key. ADR 021 records the contract; the stored row format is fixed
because collections already on disk were written in it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self, TypedDict

from pydantic import BaseModel, ConfigDict, ValidationError

CHUNK_CONTRACT_VERSION = "1"

StoredValue = str | int | float | bool


class ChunkField(StrEnum):
    """Stored metadata keys that vector-store queries filter on."""

    SOURCE = "source"
    HEADING_PATH = "heading_path"


@dataclass(frozen=True)
class ChunkDraft:
    """A piece of text a chunking strategy produced, before it is placed in its source.

    ``oversized`` is set only by strategies that can emit an atomic value larger
    than the configured limit; ``None`` means the strategy does not track it.
    """

    text: str
    heading_path: str = ""
    chunk_type: str = "text_block"
    oversized: bool | None = None


@dataclass(frozen=True)
class Chunk:
    """A chunk draft placed in its source, with a deterministic chunk ID.

    Offsets are absent: strategies strip whitespace and structural chunking
    repacks blocks, so source offsets would imply precision they do not have.
    """

    text: str
    source: str
    chunk_index: int
    chunking_strategy: str
    chunk_id: str
    heading_path: str = ""
    chunk_type: str = "text_block"
    oversized: bool | None = None

    @classmethod
    def place(cls, draft: ChunkDraft, *, source: str, chunk_index: int, strategy: str) -> Self:
        # The ID depends only on content and position, never on time or process
        # order, so re-ingesting unchanged input with the same settings rewrites
        # the same IDs; duplicate text stays distinct through its index.
        identity = "\0".join(
            (CHUNK_CONTRACT_VERSION, source, strategy, str(chunk_index), draft.text)
        )
        return cls(
            text=draft.text,
            source=source,
            chunk_index=chunk_index,
            chunking_strategy=strategy,
            chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            heading_path=draft.heading_path,
            chunk_type=draft.chunk_type,
            oversized=draft.oversized,
        )


@dataclass(frozen=True)
class SourceProvenance:
    """What was true of a source when it was ingested; shared by all its chunks."""

    file_type: str
    ingested_at: str
    content_hash: str
    source_mtime: float
    git_commit: str
    tenant_id: str


class ChunkMetadata(BaseModel):
    """The metadata row stored beside each chunk's text and embedding.

    Writes are complete: ``of`` fills every field. Reads are tolerant, because
    collections hold rows from older writers and plugins return arbitrary
    metadata: a missing, ill-typed, or unknown key never raises, it reads as the
    field's default. ``None`` fields are omitted when stored, since Chroma either
    rejects a ``None`` value or silently drops the key depending on the call.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    oversized: bool | None = None
    chunking_strategy: str = ""
    source: str | None = None
    file_type: str = ""
    chunk_index: int | None = None
    chunk_id: str = ""
    heading_path: str = ""
    chunk_type: str = ""
    ingested_at: str | None = None
    content_hash: str = ""
    source_mtime: float | None = None
    git_commit: str = ""
    tenant_id: str = ""

    @classmethod
    def of(cls, chunk: Chunk, provenance: SourceProvenance) -> Self:
        return cls(
            oversized=chunk.oversized,
            chunking_strategy=chunk.chunking_strategy,
            source=chunk.source,
            file_type=provenance.file_type,
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
            heading_path=chunk.heading_path,
            chunk_type=chunk.chunk_type,
            ingested_at=provenance.ingested_at,
            content_hash=provenance.content_hash,
            source_mtime=provenance.source_mtime,
            git_commit=provenance.git_commit,
            tenant_id=provenance.tenant_id,
        )

    @classmethod
    def from_stored(cls, stored: Mapping[str, object]) -> Self:
        # Rows almost always validate, so the common path stays a single
        # validation; only a row with ill-typed keys pays for a second pass.
        try:
            return cls.model_validate(stored)
        except ValidationError as error:
            invalid = {str(detail["loc"][0]) for detail in error.errors() if detail["loc"]}
            return cls.model_validate({k: v for k, v in stored.items() if k not in invalid})

    def to_stored(self) -> dict[str, StoredValue]:
        return self.model_dump(exclude_none=True)


class RetrievalContext(TypedDict, total=False):
    """A chunk returned for a question: its text, stored metadata, and ranking score.

    This is also the shape third-party retriever plugins return, so its keys
    are part of the plugin contract.
    """

    text: str
    chunk_id: str
    source: str
    chunk_index: int
    score: float
    distance: float
    ingested_at: str | None
    metadata: dict[str, Any]
    freshness_factor: float


def retrieval_context(
    text: str,
    stored: Mapping[str, Any],
    *,
    score: float,
    distance: float | None = None,
    chunk_id: str | None = None,
) -> RetrievalContext:
    """Build the retrieval context for one stored chunk.

    ``chunk_id`` is the vector-store ID when the caller has one; otherwise the
    ID recorded in the stored row is used, and the key is left out when neither
    exists.
    """
    metadata = ChunkMetadata.from_stored(stored)
    context: RetrievalContext = {
        "text": text,
        "source": "unknown" if metadata.source is None else metadata.source,
        "chunk_index": -1 if metadata.chunk_index is None else metadata.chunk_index,
        "score": score,
        "ingested_at": metadata.ingested_at,
        "metadata": dict(stored),
    }
    resolved_id = chunk_id or metadata.chunk_id
    if resolved_id:
        context["chunk_id"] = resolved_id
    if distance is not None:
        context["distance"] = distance
    return context

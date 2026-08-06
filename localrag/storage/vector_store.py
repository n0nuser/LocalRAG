from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha1
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from localrag.embedding.base import EmbeddingIncompatibilityError, EmbeddingProvider

logger = logging.getLogger(__name__)

# Used only when the client cannot report its own cap. Below every Chroma limit
# seen in practice, so a wrong guess costs an extra round trip, not a failure.
_DEFAULT_MAX_BATCH_SIZE = 1000


@dataclass
class VectorStore:
    client: chromadb.ClientAPI
    collection: Collection
    persist_path: Path | None = None
    _write_lock: threading.RLock = dataclass_field(
        default_factory=threading.RLock, init=False, repr=False
    )

    @classmethod
    def create(cls, persist_path: str, collection_name: str) -> VectorStore:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_ready persist_path=%s collection=%s",
            persist_path,
            collection_name,
        )
        return cls(client=client, collection=collection, persist_path=Path(persist_path))

    @classmethod
    def open(cls, persist_path: str, collection_name: str) -> VectorStore:
        """Open an existing collection without creating or mutating it."""
        client = chromadb.PersistentClient(path=persist_path)
        return cls(
            client=client,
            collection=client.get_collection(name=collection_name),
            persist_path=Path(persist_path),
        )

    def for_collection(self, name: str) -> VectorStore:
        """Open another collection on this client's persist path."""
        return type(self)(
            client=self.client,
            collection=self.client.get_collection(name=name),
            persist_path=self.persist_path,
        )

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            logger.error(
                "vector_upsert_length_mismatch source=%s chunks=%s embeddings=%s metadatas=%s",
                source,
                len(chunks),
                len(embeddings),
                len(metadatas),
            )
            raise ValueError("chunks, embeddings, and metadatas must have the same length")
        if any(len(emb) == 0 for emb in embeddings):
            logger.error("vector_upsert_empty_embedding source=%s", source)
            raise ValueError("embeddings must be non-empty vectors")

        with self._write_lock:
            self._upsert(source, chunks, embeddings, metadatas)
            self._bump_revision()
        logger.debug(
            "vector_upsert source=%s chunk_count=%s",
            source,
            len(chunks),
        )

    def replace_source(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Replace one source while readers see either the old or new version."""
        if len(chunks) != len(embeddings) or len(chunks) != len(metadatas):
            raise ValueError("chunks, embeddings, and metadatas must have the same length")
        if any(len(embedding) == 0 for embedding in embeddings):
            raise ValueError("embeddings must be non-empty vectors")
        with self._write_lock:
            old = self.collection.get(
                where={"source": source},
                include=["documents", "metadatas", "embeddings"],
            )
            try:
                self._upsert(source, chunks, embeddings, metadatas)
                new_ids = {
                    str(metadata.get("chunk_id") or self._chunk_id(source, index))
                    for index, metadata in enumerate(metadatas)
                }
                old_id_set = {str(chunk_id) for chunk_id in old.get("ids") or []}
                obsolete = old_id_set - new_ids
                if obsolete:
                    self.collection.delete(ids=sorted(obsolete))
                self._bump_revision()
            except Exception:
                # Chroma has no multi-operation transaction. Restore the complete
                # previous source so a failed replacement cannot destroy it.
                try:
                    self.collection.delete(where={"source": source})
                    old_ids = old.get("ids") or []
                    old_documents = old.get("documents") or []
                    old_embeddings = old.get("embeddings")
                    old_metadatas = old.get("metadatas") or []
                    if old_ids:
                        # Batched too: restoring a source large enough to trip the
                        # cap must not fail the rollback itself.
                        # Chroma types these more loosely than the write path
                        # accepts; the values are already the shape it returned.
                        self._upsert_batched(
                            ids=[str(chunk_id) for chunk_id in old_ids],
                            documents=[str(document) for document in old_documents],
                            embeddings=[
                                [float(value) for value in embedding]
                                for embedding in (
                                    old_embeddings if old_embeddings is not None else []
                                )
                            ],
                            metadatas=[dict(metadata) for metadata in old_metadatas],
                        )
                except Exception:
                    logger.exception("vector_replace_rollback_failed source=%s", source)
                raise

    def _upsert(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ids = [
            str(metadata.get("chunk_id") or self._chunk_id(source=source, chunk_index=index))
            for index, metadata in enumerate(metadatas)
        ]
        self._upsert_batched(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    def _upsert_batched(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Write in backend-sized batches.

        Chroma rejects a batch larger than its own cap, which would fail the whole
        document after every embedding has already been paid for.
        """
        batch_size = self._max_batch_size()
        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            self.collection.upsert(
                ids=ids[start:stop],
                documents=documents[start:stop],
                embeddings=embeddings[start:stop],  # type: ignore[arg-type]
                metadatas=metadatas[start:stop],  # type: ignore[arg-type]
            )

    def _max_batch_size(self) -> int:
        """Return the backend's write cap, falling back to a conservative default."""
        get_limit = getattr(self.client, "get_max_batch_size", None)
        if get_limit is None:
            return _DEFAULT_MAX_BATCH_SIZE
        try:
            limit = int(get_limit())
        except Exception:
            logger.debug("vector_max_batch_size_unavailable", exc_info=True)
            return _DEFAULT_MAX_BATCH_SIZE
        return max(1, limit)

    def _source_ids(self, source: str) -> set[str]:
        raw = self.collection.get(where={"source": source}, include=[])
        return {str(chunk_id) for chunk_id in raw.get("ids") or []}

    def ensure_embedding_compatibility(
        self, provider: EmbeddingProvider, dimension: int | None = None, model: str | None = None
    ) -> None:
        """Reject a collection whose recorded embedding space differs from runtime."""
        metadata = dict(self.collection.metadata or {})
        expected = {
            "localrag:embedding_provider": provider.provider_name,
            "localrag:embedding_model": model or provider.model,
        }
        recorded_dimension = metadata.get("localrag:embedding_dimension")
        if recorded_dimension is not None:
            try:
                recorded_dimension = int(recorded_dimension)
            except (TypeError, ValueError) as exc:
                raise EmbeddingIncompatibilityError(
                    "Collection embedding metadata is malformed"
                ) from exc
        effective_dimension = dimension or provider.dimension
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise EmbeddingIncompatibilityError(
                    "Embedding provider/model is incompatible with the collection; "
                    "run `localrag collections rebuild` after selecting the new embedding space."
                )
        if (
            recorded_dimension is not None
            and effective_dimension is not None
            and recorded_dimension != effective_dimension
        ):
            raise EmbeddingIncompatibilityError(
                "Embedding dimension is incompatible with the collection; "
                "run `localrag collections rebuild`."
            )
        if recorded_dimension is None and effective_dimension is not None:
            raw = self.collection.get(include=["embeddings"], limit=1)
            rows = raw.get("embeddings")
            if rows is not None and len(rows) > 0 and len(rows[0]) != effective_dimension:
                raise EmbeddingIncompatibilityError(
                    "Embedding dimension is incompatible with the legacy collection; "
                    "run `localrag collections rebuild`."
                )
        if effective_dimension is None and recorded_dimension is not None:
            return

    def record_embedding_compatibility(
        self, provider: EmbeddingProvider, dimension: int, model: str | None = None
    ) -> None:
        """Record the embedding identity after vectors have been validated."""
        self.ensure_embedding_compatibility(provider, dimension, model)
        metadata = dict(self.collection.metadata or {})
        metadata.update(
            {
                "localrag:embedding_provider": provider.provider_name,
                "localrag:embedding_model": model or provider.model,
                "localrag:embedding_dimension": dimension,
            }
        )
        # Chroma treats hnsw:space as immutable and rejects it in modify().
        metadata.pop("hnsw:space", None)
        self.collection.modify(metadata=metadata)

    def delete_by_source(self, source: str) -> None:
        with self._write_lock:
            self.collection.delete(where={"source": source})
            self._bump_revision()
        logger.debug("vector_delete_by_source source=%s", source)

    def _bump_revision(self) -> None:
        metadata = dict(getattr(self.collection, "metadata", None) or {})
        metadata["localrag:corpus_revision"] = int(metadata.get("localrag:corpus_revision", 0)) + 1
        modify = getattr(self.collection, "modify", None)
        if modify is not None:
            metadata.pop("hnsw:space", None)
            modify(metadata=metadata)

    def query(
        self, embedding: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        logger.debug("vector_query top_k=%s where=%s", top_k, where)
        with self._write_lock:
            return self.collection.query(  # type: ignore[return-value]
                query_embeddings=[embedding],  # type: ignore[arg-type]
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

    def get_chunks_by_headings(
        self,
        headings: list[tuple[str, str]],
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[tuple[str, str], list[tuple[int, str]]]:
        """Fetch sibling chunks for multiple sections in one collection lookup."""
        if not headings:
            return {}

        clauses = [
            {"$and": [{"source": source}, {"heading_path": heading_path}]}
            for source, heading_path in headings
        ]
        where: dict[str, Any] = {"$or": clauses} if len(clauses) > 1 else clauses[0]
        if metadata_filter:
            where = {"$and": [where, *[{key: value} for key, value in metadata_filter.items()]]}
        with self._write_lock:
            raw = self.collection.get(where=where, include=["documents", "metadatas"])
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        requested = set(headings)
        grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for document, metadata in zip(documents, metadatas, strict=False):
            if not isinstance(document, str):
                continue
            metadata_map = metadata if isinstance(metadata, dict) else {}
            key = (str(metadata_map.get("source", "")), str(metadata_map.get("heading_path", "")))
            if key not in requested or any(
                metadata_map.get(filter_key) != filter_value
                for filter_key, filter_value in (metadata_filter or {}).items()
            ):
                continue
            chunk_index = metadata_map.get("chunk_index")
            if isinstance(chunk_index, int):
                grouped.setdefault(key, []).append((chunk_index, document))
        for pairs in grouped.values():
            pairs.sort(key=lambda pair: pair[0])
        return grouped

    def get_chunks_by_heading(self, source: str, heading_path: str) -> list[tuple[int, str]]:
        """Fetch sibling chunks for one section, preserving the legacy helper contract."""
        return self.get_chunks_by_headings([(source, heading_path)]).get((source, heading_path), [])

    def list_distinct_sources(self) -> list[str]:
        with self._write_lock:
            raw = self.collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas")
        if not metadatas:
            return []
        sources: set[str] = set()
        for md in metadatas:
            if md and isinstance(md, dict) and "source" in md:
                sources.add(str(md["source"]))
        return sorted(sources)

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]

    def delete_collection(self, name: str) -> None:
        with self._write_lock:
            segment_ids = self._persisted_segment_ids(name)
            self.client.delete_collection(name)
            for segment_id in segment_ids:
                segment_path = (self.persist_path / segment_id) if self.persist_path else None
                if segment_path is None:
                    continue
                try:
                    _remove_directory(segment_path)
                except OSError:
                    logger.warning("vector_collection_segment_cleanup_failed path=%s", segment_path)
            if getattr(self.collection, "name", None) == name:
                self.collection = self.client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
        logger.warning("vector_collection_deleted name=%s", name)

    def _persisted_segment_ids(self, name: str) -> list[str]:
        """Read HNSW segment IDs before Chroma removes their collection metadata."""
        if self.persist_path is None:
            return []
        database = self.persist_path / "chroma.sqlite3"
        if not database.is_file():
            return []
        try:
            collection_id = str(self.client.get_collection(name=name).id)
            with sqlite3.connect(database) as connection:
                rows = connection.execute(
                    "SELECT id FROM segments WHERE collection = ? "
                    "AND type = 'urn:chroma:segment/vector/hnsw-local-persisted'",
                    (collection_id,),
                ).fetchall()
        except (OSError, sqlite3.Error):
            logger.warning("vector_collection_segment_lookup_failed name=%s", name, exc_info=True)
            return []
        return [str(row[0]) for row in rows]

    def get_all_chunks(self) -> list[tuple[str, str, dict[str, Any]]]:
        with self._write_lock:
            raw = self.collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        all_chunks: list[tuple[str, str, dict[str, Any]]] = []
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=False):
            if not isinstance(chunk_id, str) or not isinstance(document, str):
                continue
            normalized_metadata = metadata if isinstance(metadata, dict) else {}
            all_chunks.append((chunk_id, document, normalized_metadata))
        return all_chunks

    @staticmethod
    def _chunk_id(source: str, chunk_index: int) -> str:
        return sha1(f"{source}:{chunk_index}".encode(), usedforsecurity=False).hexdigest()


def _remove_directory(path: Path) -> None:
    """Remove one Chroma segment without touching unrelated persist-path entries."""
    if path.is_dir():
        shutil.rmtree(path)

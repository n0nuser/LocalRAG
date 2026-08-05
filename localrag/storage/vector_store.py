from __future__ import annotations

import logging
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


@dataclass
class VectorStore:
    client: chromadb.ClientAPI
    collection: Collection
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
        return cls(client=client, collection=collection)

    @classmethod
    def open(cls, persist_path: str, collection_name: str) -> VectorStore:
        """Open an existing collection without creating or mutating it."""
        client = chromadb.PersistentClient(path=persist_path)
        return cls(client=client, collection=client.get_collection(name=collection_name))

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
                        self.collection.upsert(
                            ids=old_ids,
                            documents=old_documents,
                            embeddings=(old_embeddings if old_embeddings is not None else []),  # type: ignore[arg-type]
                            metadatas=old_metadatas,  # type: ignore[arg-type]
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
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )

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
            self.client.delete_collection(name)
            if getattr(self.collection, "name", None) == name:
                self.collection = self.client.get_or_create_collection(
                    name=name, metadata={"hnsw:space": "cosine"}
                )
        logger.warning("vector_collection_deleted name=%s", name)

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

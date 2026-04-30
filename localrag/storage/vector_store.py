from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


@dataclass
class VectorStore:
    client: chromadb.ClientAPI
    collection: Collection

    @classmethod
    def create(cls, persist_path: str, collection_name: str) -> VectorStore:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
        collection = client.get_or_create_collection(name=collection_name)
        logger.info(
            "vector_store_ready persist_path=%s collection=%s",
            persist_path,
            collection_name,
        )
        return cls(client=client, collection=collection)

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

        ids = [self._chunk_id(source=source, chunk_index=index) for index in range(len(chunks))]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        logger.debug(
            "vector_upsert source=%s chunk_count=%s",
            source,
            len(chunks),
        )

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
        logger.debug("vector_delete_by_source source=%s", source)

    def query(self, embedding: list[float], top_k: int) -> dict[str, Any]:
        logger.debug("vector_query top_k=%s", top_k)
        return self.collection.query(  # type: ignore[return-value]
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def list_distinct_sources(self) -> list[str]:
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
        self.client.delete_collection(name)
        logger.warning("vector_collection_deleted name=%s", name)

    @staticmethod
    def _chunk_id(source: str, chunk_index: int) -> str:
        return sha1(f"{source}:{chunk_index}".encode(), usedforsecurity=False).hexdigest()

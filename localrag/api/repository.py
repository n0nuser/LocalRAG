from __future__ import annotations

from dataclasses import dataclass

from localrag.storage.vector_store import VectorStore


@dataclass(frozen=True)
class ChromaCollectionRepository:
    """Persistence for Chroma collection names (list/delete) used by the HTTP API."""

    _vector_store: VectorStore

    def list_collection_names(self) -> list[str]:
        return self._vector_store.list_collections()

    def delete_collection(self, name: str) -> None:
        self._vector_store.delete_collection(name)

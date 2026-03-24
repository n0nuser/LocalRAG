from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection


@dataclass
class VectorStore:
    client: chromadb.ClientAPI
    collection: Collection

    @classmethod
    def create(cls, persist_path: str, collection_name: str) -> VectorStore:
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=persist_path)
        collection = client.get_or_create_collection(name=collection_name)
        return cls(client=client, collection=collection)

    def add_chunks(
        self,
        source: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        ids = [self._chunk_id(source=source, chunk_index=index) for index in range(len(chunks))]
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def query(self, embedding: list[float], top_k: int) -> dict[str, Any]:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def list_collections(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for collection in self.client.list_collections():
            count = self.client.get_or_create_collection(collection.name).count()
            result.append({"name": collection.name, "count": count})
        return result

    def delete_collection(self, name: str) -> None:
        self.client.delete_collection(name)

    @staticmethod
    def _chunk_id(source: str, chunk_index: int) -> str:
        return sha1(f"{source}:{chunk_index}".encode(), usedforsecurity=False).hexdigest()

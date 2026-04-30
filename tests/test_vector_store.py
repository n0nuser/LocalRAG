from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import pytest

import localrag.storage.vector_store as vector_store_module
from localrag.storage.vector_store import VectorStore


@dataclass
class FakeCollection:
    upsert_calls: list[dict[str, object]]
    delete_calls: list[dict[str, object]]
    query_calls: list[dict[str, object]]
    query_result: dict[str, object]
    get_return: dict[str, object] = field(default_factory=dict)

    def get(
        self,
        ids: list[str] | None = None,
        include: list[str] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = (ids, kwargs)
        _ = include
        return self.get_return

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
    ) -> None:
        self.upsert_calls.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        )

    def delete(self, where: dict[str, object]) -> None:
        self.delete_calls.append({"where": where})

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, object]:
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include,
            }
        )
        return self.query_result


@dataclass
class FakeClient:
    collections: list[object]
    deleted_collections: list[str]

    def list_collections(self) -> list[object]:
        return self.collections

    def get_or_create_collection(self, name: str) -> object:
        # The real Chroma returns Collection; we only need .count().
        for c in self.collections:
            if getattr(c, "name", None) == name:
                return c
        message = f"Missing collection {name}"
        raise AssertionError(message)

    def delete_collection(self, name: str) -> None:
        self.deleted_collections.append(name)


@dataclass
class FakeNameCollection:
    name: str
    count_value: int

    def count(self) -> int:
        return self.count_value


def test_vector_store_add_chunks_validates_lengths() -> None:
    collection = FakeCollection(
        upsert_calls=[],
        delete_calls=[],
        query_calls=[],
        query_result={},
        get_return={},
    )
    client = FakeClient(collections=[], deleted_collections=[])
    store = VectorStore(client=client, collection=collection)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="same length"):
        store.add_chunks(
            source="s",
            chunks=["a", "b"],
            embeddings=[[0.1]],
            metadatas=[{"x": 1}, {"y": 2}],
        )


def test_vector_store_add_chunks_rejects_empty_embeddings() -> None:
    collection = FakeCollection(
        upsert_calls=[],
        delete_calls=[],
        query_calls=[],
        query_result={},
        get_return={},
    )
    client = FakeClient(collections=[], deleted_collections=[])
    store = VectorStore(client=client, collection=collection)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        store.add_chunks(
            source="s",
            chunks=["a"],
            embeddings=[[]],
            metadatas=[{"x": 1}],
        )


def test_vector_store_upsert_delete_query_and_list_collections() -> None:
    collection = FakeCollection(
        upsert_calls=[],
        delete_calls=[],
        query_calls=[],
        query_result={
            "documents": [["chunk-a"]],
            "metadatas": [[{"source": "s", "chunk_index": 0}]],
            "distances": [[0.12]],
        },
        get_return={},
    )

    c1 = FakeNameCollection(name="col-1", count_value=2)
    c2 = FakeNameCollection(name="col-2", count_value=0)
    client = FakeClient(collections=[c1, c2], deleted_collections=[])
    store = VectorStore(client=client, collection=collection)  # type: ignore[arg-type]

    source = "src"
    chunks = ["c0", "c1"]
    embeddings = [[0.0], [1.0]]
    metadatas = [
        {"source": source, "file_type": ".md", "chunk_index": 0},
        {"source": source, "file_type": ".md", "chunk_index": 1},
    ]

    store.add_chunks(source=source, chunks=chunks, embeddings=embeddings, metadatas=metadatas)
    assert len(collection.upsert_calls) == 1

    ids = collection.upsert_calls[0]["ids"]  # type: ignore[index]
    expected_ids = [
        sha1(f"{source}:{chunk_index}".encode(), usedforsecurity=False).hexdigest()
        for chunk_index in range(2)
    ]
    assert ids == expected_ids

    store.delete_by_source(source=source)
    assert collection.delete_calls == [{"where": {"source": source}}]

    out = store.query(embedding=[0.1], top_k=3)
    assert out == collection.query_result
    assert collection.query_calls[0]["n_results"] == 3  # type: ignore[index]

    collections = store.list_collections()
    assert collections == ["col-1", "col-2"]

    store.delete_collection("col-1")
    assert client.deleted_collections == ["col-1"]


def test_vector_store_list_distinct_sources() -> None:
    collection = FakeCollection(
        upsert_calls=[],
        delete_calls=[],
        query_calls=[],
        query_result={},
        get_return={
            "metadatas": [
                {"source": "/b.md", "chunk_index": 0},
                {"source": "/a.md", "chunk_index": 0},
                {"source": "/a.md", "chunk_index": 1},
            ],
        },
    )
    client = FakeClient(collections=[], deleted_collections=[])
    store = VectorStore(client=client, collection=collection)  # type: ignore[arg-type]

    assert store.list_distinct_sources() == ["/a.md", "/b.md"]


def test_vector_store_create_initializes_persistent_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created_dir = tmp_path / "chroma"
    collection_name = "localrag"
    sentinel_collection = object()

    class FakeClient:
        def __init__(self, path: str) -> None:
            self.path = path
            self.got_collection_name: str | None = None

        def get_or_create_collection(self, name: str) -> object:
            self.got_collection_name = name
            return sentinel_collection

    def fake_persistent_client(path: str) -> FakeClient:
        return FakeClient(path=path)

    monkeypatch.setattr(vector_store_module.chromadb, "PersistentClient", fake_persistent_client)

    store = VectorStore.create(persist_path=str(created_dir), collection_name=collection_name)

    assert created_dir.exists()
    assert store.collection is sentinel_collection

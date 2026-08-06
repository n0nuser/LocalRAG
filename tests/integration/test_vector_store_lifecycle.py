from __future__ import annotations

from pathlib import Path

import pytest

from localrag.storage.vector_store import VectorStore

pytestmark = pytest.mark.integration


def test_real_chroma_delete_reclaims_hnsw_segment(tmp_path: Path) -> None:
    persist_path = tmp_path / "chroma"
    store = VectorStore.create(str(persist_path), "lifecycle")
    count = 1001
    store.add_chunks(
        source="fixture",
        chunks=["fixture text"] * count,
        embeddings=[[1.0, 0.0, 0.0]] * count,
        metadatas=[{"chunk_id": str(index), "source": "fixture"} for index in range(count)],
    )
    segment_directories = [path for path in persist_path.iterdir() if path.is_dir()]
    assert segment_directories

    store.delete_collection("lifecycle")

    assert not any(path.exists() for path in segment_directories)

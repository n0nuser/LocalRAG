from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier, Thread

from localrag.embedding.cache import EmbeddingCache


@dataclass
class CountingProvider:
    provider_name: str = "fake"
    model: str = "model-v1"
    model_revision: str = "digest-a"
    dimension: int | None = 2
    timeout_seconds: float = 1
    calls: int = 0

    def embed_batch(
        self, texts: Sequence[str], *, batch_size: int | None = None, model: str | None = None
    ) -> list[list[float]]:
        _ = batch_size
        self.calls += 1
        return [
            [float(len(text) + index) for index in range(self.dimension or 0)] for text in texts
        ]


def test_cache_warm_hit_preserves_duplicates_and_metadata_is_not_stored(tmp_path: Path) -> None:
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path)

    assert cache.embed_batch(provider, ["same", "same"]) == [[4.0, 5.0], [4.0, 5.0]]
    assert cache.embed_batch(provider, ["same"]) == [[4.0, 5.0]]
    assert provider.calls == 1
    assert all("same" not in entry.read_text() for entry in tmp_path.glob("*.json"))


def test_cache_identity_invalidates_model_revision_preprocessing_and_dimension(
    tmp_path: Path,
) -> None:
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path, preprocessing_version="1")
    cache.embed_batch(provider, ["text"])

    provider.model_revision = "digest-b"
    cache.embed_batch(provider, ["text"])
    provider.dimension = 3
    cache.embed_batch(provider, ["text"])
    assert provider.calls == 3


def test_corrupt_entry_is_deleted_and_recomputed(tmp_path: Path) -> None:
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path)
    cache.embed_batch(provider, ["text"])
    entry = next(tmp_path.glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")

    assert cache.embed_batch(provider, ["text"]) == [[4.0, 5.0]]
    assert provider.calls == 2


def test_concurrent_threads_share_one_atomic_miss(tmp_path: Path) -> None:
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path)
    barrier = Barrier(2)
    results: list[list[list[float]]] = []

    def run() -> None:
        barrier.wait()
        results.append(cache.embed_batch(provider, ["text"]))

    threads = [Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [[[4.0, 5.0]], [[4.0, 5.0]]]
    assert provider.calls == 1


def test_cache_is_always_active(tmp_path: Path) -> None:
    """Caching has no opt-out flag: a repeated text must not re-hit the provider."""
    provider = CountingProvider()
    cache = EmbeddingCache(tmp_path)
    first = cache.embed_batch(provider, ["text"])
    second = cache.embed_batch(provider, ["text"])
    assert first == second
    assert provider.calls == 1
    assert list(tmp_path.glob("*.json"))

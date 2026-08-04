"""In-process TTL cache for repeated/near-identical queries (no external service)."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from cachetools import TTLCache


def make_cache_key(
    question: str,
    model: str | None,
    n_results: int | None,
    retrieval_mode: str,
    *,
    metadata_filter: dict[str, Any] | None = None,
    collection: str = "",
    provider: str = "",
    corpus_revision: str = "",
) -> str:
    payload = json.dumps(
        {
            "question": question.strip().lower(),
            "model": model,
            "n_results": n_results,
            "retrieval_mode": retrieval_mode,
            "metadata_filter": metadata_filter or {},
            "collection": collection,
            "provider": provider,
            "corpus_revision": corpus_revision,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class QueryCache:
    def __init__(self, maxsize: int, ttl_seconds: float) -> None:
        self._enabled = ttl_seconds > 0 and maxsize > 0
        self._cache: TTLCache[str, dict[str, Any]] = TTLCache(
            maxsize=max(1, maxsize), ttl=max(0.001, ttl_seconds)
        )
        self._lock = threading.RLock()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._cache[key] = value

    def clear(self) -> None:
        """Invalidate all entries after a corpus or collection mutation."""
        with self._lock:
            self._cache.clear()

from __future__ import annotations

import time

from localrag.rag.query_cache import QueryCache, make_cache_key


def test_query_cache_returns_none_when_disabled() -> None:
    cache = QueryCache(maxsize=10, ttl_seconds=0)
    key = make_cache_key("q", None, None, "hybrid")
    cache.set(key, {"answer": "x"})
    assert cache.get(key) is None


def test_query_cache_hits_within_ttl_and_expires_after() -> None:
    cache = QueryCache(maxsize=10, ttl_seconds=0.05)
    key = make_cache_key("q", "m", 5, "hybrid")
    cache.set(key, {"answer": "cached"})

    assert cache.get(key) == {"answer": "cached"}
    time.sleep(0.1)
    assert cache.get(key) is None


def test_make_cache_key_is_case_and_whitespace_insensitive() -> None:
    key_a = make_cache_key("  What is X? ", "m", 5, "hybrid")
    key_b = make_cache_key("what is x?", "m", 5, "hybrid")
    assert key_a == key_b


def test_make_cache_key_differs_by_model() -> None:
    key_a = make_cache_key("q", "model-a", 5, "hybrid")
    key_b = make_cache_key("q", "model-b", 5, "hybrid")
    assert key_a != key_b

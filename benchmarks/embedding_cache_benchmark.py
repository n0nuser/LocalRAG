"""Measure cold and warm ingestion-boundary embedding calls.

Example: ``uv run python benchmarks/embedding_cache_benchmark.py corpus/*.md``.
The provider must be available locally (Ollama by default).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from statistics import quantiles
from typing import Any

from localrag.embedding.base import EmbeddingProvider
from localrag.embedding.cache import EmbeddingCache
from localrag.embedding.factory import build_embedding_provider
from localrag.settings import Settings


class CountingProvider:
    def __init__(self, provider: EmbeddingProvider) -> None:
        self.provider = provider
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)

    def embed_batch(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls += 1
        return self.provider.embed_batch(texts, **kwargs)


def _percentiles(values: list[float]) -> dict[str, float]:
    if len(values) < 2:
        return {"p50_ms": values[0] * 1000, "p95_ms": values[0] * 1000}
    points = quantiles(values, n=100)
    return {"p50_ms": points[49] * 1000, "p95_ms": points[94] * 1000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="+", type=Path)
    parser.add_argument("--cache-path", type=Path, default=Path("./data/embedding-cache-benchmark"))
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    texts = [path.read_text(encoding="utf-8") for path in sorted(args.corpus)]
    settings = Settings(
        embedding_provider=args.provider,
        embedding_model=args.model,
    )
    provider = CountingProvider(build_embedding_provider(settings))
    cache = EmbeddingCache(args.cache_path)
    cache.clear()
    measurements: list[dict[str, object]] = []
    for label in ("cold", "warm"):
        started = time.perf_counter()
        before_calls = provider.calls
        for text in texts:
            cache.embed_batch(provider, [text], model=args.model or None)
        elapsed = time.perf_counter() - started
        measurements.append(
            {
                "run": label,
                "provider_calls": provider.calls - before_calls,
                "wall_time_ms": elapsed * 1000,
                **_percentiles([elapsed / max(len(texts), 1)]),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
            }
        )
    cache_bytes = sum(path.stat().st_size for path in args.cache_path.glob("*.json"))
    sys.stdout.write(
        json.dumps(
            {
                "corpus": [str(path) for path in sorted(args.corpus)],
                "provider": provider.provider_name,
                "model": args.model or provider.model,
                "model_revision": getattr(provider, "model_revision", ""),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cache_path": str(args.cache_path),
                "cache_bytes": cache_bytes,
                "measurements": measurements,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

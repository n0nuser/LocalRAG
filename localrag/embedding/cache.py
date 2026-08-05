"""Safe, provider-aware disk cache for ingestion embedding vectors."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from localrag.embedding.base import EmbeddingProvider, validate_vectors

try:
    import fcntl
except ImportError:  # pragma: no cover - LocalRAG currently supports Unix hosts.
    fcntl = None  # type: ignore[assignment]

CACHE_SCHEMA_VERSION = 1
_LOCK_TIMEOUT_SECONDS = 5.0


class EmbeddingCache:
    """Persistent cache of vectors only; cache failures always fall back to provider calls."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 10_000,
        max_bytes: int = 1_000_000_000,
        preprocessing_version: str = "1",
        task_prefix: str = "",
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.preprocessing_version = preprocessing_version
        self.task_prefix = task_prefix
        self._thread_lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.write_failures = 0

    def embed_batch(
        self,
        provider: EmbeddingProvider,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        model: str | None = None,
    ) -> list[list[float]]:
        """Read or compute vectors in input order, including duplicate input text."""
        if not texts:
            return []

        with self._exclusive_lock():
            dimension = self._known_dimension(provider)
            keys = [self._key(provider, text, model=model, dimension=dimension) for text in texts]
            cached: list[list[float] | None] = [self._read(key) for key in keys]
            missing = [index for index, vector in enumerate(cached) if vector is None]
            self.hits += len(texts) - len(missing)
            self.misses += len(missing)
            if missing:
                computed = self._provider_embed(
                    provider,
                    [texts[index] for index in missing],
                    batch_size=batch_size,
                    model=model,
                )
                actual_dimension = len(computed[0])
                for index, vector in zip(missing, computed, strict=True):
                    if len(vector) != actual_dimension:
                        raise ValueError("provider returned inconsistent embedding dimensions")
                    computed_key = self._key(
                        provider, texts[index], model=model, dimension=actual_dimension
                    )
                    self._write(computed_key, vector)
                    cached[index] = vector
            result = [vector for vector in cached if vector is not None]
            if len(result) != len(texts):
                raise ValueError("embedding cache returned an incomplete result")
            return result

    def clear(self) -> None:
        """Remove all cache entries without touching any vector-store metadata."""
        if not self.path.exists():
            return
        with self._exclusive_lock():
            for entry in self.path.glob("*.json"):
                try:
                    entry.unlink()
                except OSError:
                    continue

    def _provider_embed(
        self,
        provider: EmbeddingProvider,
        texts: Sequence[str],
        *,
        batch_size: int | None,
        model: str | None,
    ) -> list[list[float]]:
        embed_batch = getattr(provider, "embed_batch", None)
        if embed_batch is not None:
            vectors = embed_batch(texts, batch_size=batch_size, model=model)
        else:
            legacy_provider = provider  # compatibility with integrations using the old seam
            vectors = legacy_provider.embed_texts(  # type: ignore[attr-defined]
                list(texts), batch_size or len(texts), model=model
            )
        expected_dimension = self._known_dimension(provider)
        if expected_dimension is not None and any(
            len(vector) != expected_dimension for vector in vectors
        ):
            raise ValueError("provider returned an incompatible embedding dimension")
        return vectors

    def _known_dimension(self, provider: EmbeddingProvider) -> int | None:
        dimension = getattr(provider, "dimension", None)
        return dimension if isinstance(dimension, int) and dimension > 0 else None

    def _key(
        self,
        provider: EmbeddingProvider,
        text: str,
        *,
        model: str | None,
        dimension: int | None,
    ) -> str:
        effective_model = model if model is not None else provider.model
        identity = {
            "schema": CACHE_SCHEMA_VERSION,
            "scope": "ingestion",
            "provider": str(provider.provider_name).strip().lower(),
            "model": str(effective_model).strip(),
            "revision": str(getattr(provider, "model_revision", "")).strip(),
            "preprocessing": self.preprocessing_version,
            "task_prefix": self.task_prefix,
            "dimension": dimension,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _read(self, key: str) -> list[float] | None:
        entry = self.path / f"{key}.json"
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            vector = payload["vector"]
            if (
                payload["schema"] != CACHE_SCHEMA_VERSION
                or payload["key"] != key
                or not isinstance(vector, list)
                or payload["dimension"] != len(vector)
                or payload["checksum"]
                != hashlib.sha256(json.dumps(vector, separators=(",", ":")).encode()).hexdigest()
            ):
                return self._discard(entry)
            result = validate_vectors([vector], provider="cache", model=key)[0]
            os.utime(entry, None)
            return result  # noqa: TRY300
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return self._discard(entry)

    @staticmethod
    def _discard(entry: Path) -> None:
        with suppress(OSError):
            entry.unlink(missing_ok=True)

    def _write(self, key: str, vector: list[float]) -> None:
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(vector, separators=(",", ":"))
            payload = {
                "schema": CACHE_SCHEMA_VERSION,
                "key": key,
                "dimension": len(vector),
                "checksum": hashlib.sha256(serialized.encode()).hexdigest(),
                "vector": vector,
            }
            temporary = self.path / f".{key}.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(self.path / f"{key}.json")
            self._evict()
        except OSError:
            self.write_failures += 1
            with suppress(OSError, UnboundLocalError):
                temporary.unlink(missing_ok=True)

    def _evict(self) -> None:
        entries = list(self.path.glob("*.json"))
        entries.sort(key=lambda item: item.stat().st_atime)
        total_bytes = sum(item.stat().st_size for item in entries)
        while len(entries) > self.max_entries or total_bytes > self.max_bytes:
            oldest = entries.pop(0)
            size = oldest.stat().st_size
            try:
                oldest.unlink()
            except OSError:
                continue
            total_bytes -= size

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self._thread_lock:
            try:
                self.path.mkdir(parents=True, exist_ok=True)
                lock_path = self.path / ".lock"
                lock_file = lock_path.open("a+", encoding="utf-8")
            except (OSError, TimeoutError):
                # A locked, read-only, or unavailable cache is a normal miss path.
                yield
                return
            try:
                if fcntl is not None:
                    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                    while True:
                        try:
                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError("embedding cache lock timed out") from None
                            time.sleep(0.01)
            except (OSError, TimeoutError):
                # Lock acquisition errors are misses; errors from provider calls propagate.
                lock_file.close()
                yield
                return
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()

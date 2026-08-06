"""Cross-process exclusion for writers against one Chroma persist directory.

Chroma's ``PersistentClient`` keeps its HNSW segments in per-process memory with no
cross-process invalidation, so two processes writing the same persist path corrupt
each other's view (see ADR 035, which already declares multi-process writers out of
contract). This module is that external ownership boundary: an advisory ``flock`` on
``<persist_path>/.ingest.lock``.

It deliberately lives outside ``VectorStore``: the API holds an ``lru_cache``d store
for the whole process lifetime, so a store-scoped lock would block every CLI ingest
forever. Acquire it around an ingest instead, and leave read paths unlocked.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

try:
    import fcntl
except ImportError:  # pragma: no cover - LocalRAG currently supports Unix hosts.
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LOCK_FILE_NAME = ".ingest.lock"


class ConcurrentIngestError(RuntimeError):
    """Raised when another process already owns the ingest lock for a persist path."""


class _ReentrantFileLock:
    """Holds one ``flock`` per persist path and counts nested acquisitions in-process.

    ``flock`` is per-file-description, so a second acquisition inside the same process
    would silently succeed and its release would drop the lock while the outer scope
    still believed it held it. Counting keeps nesting (ingest inside rebuild) correct.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._depth: dict[Path, int] = {}
        self._files: dict[Path, TextIO] = {}

    def acquire(self, lock_path: Path) -> bool:
        """Take the lock, returning whether a release is owed. Raises only on contention."""
        with self._guard:
            if self._depth.get(lock_path, 0) > 0:
                self._depth[lock_path] += 1
                return True
            try:
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = lock_path.open("a+", encoding="utf-8")
            except OSError:
                # An unwritable persist directory cannot be ingested into anyway; let the
                # store raise the real error instead of masking it as a lock failure.
                logger.warning("ingest_lock_unavailable path=%s", lock_path)
                return False
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                descriptor.close()
                message = (
                    f"another ingest is already running against {lock_path.parent}; "
                    "wait for it to finish or use a different collection"
                )
                raise ConcurrentIngestError(message) from exc
            except OSError:
                # Filesystems without flock support (some network mounts) still ingest.
                logger.warning("ingest_lock_unsupported path=%s", lock_path)
                descriptor.close()
                return False
            self._files[lock_path] = descriptor
            self._depth[lock_path] = 1
            return True

    def release(self, lock_path: Path) -> None:
        with self._guard:
            remaining = self._depth.get(lock_path, 0) - 1
            if remaining > 0:
                self._depth[lock_path] = remaining
                return
            self._depth.pop(lock_path, None)
            descriptor = self._files.pop(lock_path, None)
            if descriptor is None:
                return
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)
            finally:
                descriptor.close()


_LOCKS = _ReentrantFileLock()


@contextmanager
def ingest_lock(persist_path: str | Path) -> Iterator[None]:
    """Own the persist path for the duration of a write, or fail fast if someone else does.

    Fails immediately rather than queueing: an ingest can run for minutes, so a caller
    blocked behind one is better told to retry than left hanging with no output.
    """
    # Resolve first: the in-process depth counter is keyed by path, so two spellings of
    # the same directory must not look like two independent locks.
    lock_path = Path(persist_path).expanduser().resolve() / LOCK_FILE_NAME
    held = _LOCKS.acquire(lock_path)
    try:
        yield
    finally:
        if held:
            _LOCKS.release(lock_path)

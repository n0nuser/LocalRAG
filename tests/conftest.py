"""Keep test runs quiet unless a test explicitly asserts on logs."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

os.environ.setdefault("LOG_LEVEL", "ERROR")

_HOLD_INGEST_LOCK = """
import sys
import time

from localrag.storage.persist_lock import ingest_lock

with ingest_lock(sys.argv[1]):
    print("acquired", flush=True)
    time.sleep(float(sys.argv[2]))
"""

IngestLockHolder = Callable[[Path, float], "subprocess.Popen[str]"]


@pytest.fixture
def ingest_lock_holder() -> Iterator[IngestLockHolder]:
    """Hold the ingest lock elsewhere; ``flock`` is per-fd, so one process cannot contend."""
    processes: list[subprocess.Popen[str]] = []

    def spawn(persist_path: Path, hold_seconds: float) -> subprocess.Popen[str]:
        process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell, test-only helper
            [sys.executable, "-c", _HOLD_INGEST_LOCK, str(persist_path), str(hold_seconds)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "acquired"
        return process

    yield spawn

    for process in processes:
        process.kill()
        process.wait(timeout=10)

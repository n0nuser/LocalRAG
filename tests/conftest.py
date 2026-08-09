"""Keep test runs quiet unless a test explicitly asserts on logs."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from localrag.settings import Settings
from localrag.settings_map import FLAT_TO_PATH, UNGROUPED_FIELDS

os.environ.setdefault("LOG_LEVEL", "ERROR")


@pytest.fixture(autouse=True)
def _isolate_configuration(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own configuration from the unit suite.

    Environment variables and ``.env`` outrank YAML (ADR 020), which is right for
    the app and wrong for tests: a real ``API_KEY`` or timeout override in the
    developer's ``.env`` silently changes what the suite is asserting on, so a
    green local run stops meaning what a green CI run means. Clearing both layers
    here makes every unit test start from the YAML/model defaults. Tests that need
    a value set it themselves, and ``monkeypatch.setenv`` in a test body still
    wins because this fixture runs first.

    Integration tests are exempt: they configure themselves from the real
    environment to reach the running stack.
    """
    if request.node.get_closest_marker("integration"):
        return

    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in (*FLAT_TO_PATH, *UNGROUPED_FIELDS):
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    # Re-assert the quiet default cleared just above; no test asserts on it.
    monkeypatch.setenv("LOG_LEVEL", "ERROR")


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

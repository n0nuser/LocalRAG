from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from localrag.storage import persist_lock
from localrag.storage.persist_lock import ConcurrentIngestError, ingest_lock

IngestLockHolder = Callable[[Path, float], "subprocess.Popen[str]"]


def test_second_holder_is_rejected_while_the_first_still_holds_the_lock(
    tmp_path: Path, ingest_lock_holder: IngestLockHolder
) -> None:
    persist_path = tmp_path / "chroma"
    ingest_lock_holder(persist_path, 30.0)

    with pytest.raises(ConcurrentIngestError) as excinfo, ingest_lock(persist_path):
        pytest.fail("the lock must not be granted twice")

    assert str(persist_path) in str(excinfo.value)


def test_lock_is_reacquirable_after_the_holder_exits(
    tmp_path: Path, ingest_lock_holder: IngestLockHolder
) -> None:
    persist_path = tmp_path / "chroma"
    ingest_lock_holder(persist_path, 0.0).wait(timeout=10)

    with ingest_lock(persist_path):
        pass

    with ingest_lock(persist_path):
        pass


def test_nested_acquisition_keeps_the_lock_until_the_outermost_scope_exits(
    tmp_path: Path, ingest_lock_holder: IngestLockHolder
) -> None:
    """Rebuild delegates to the ingest path, so the boundary must tolerate nesting."""
    persist_path = tmp_path / "chroma"

    with ingest_lock(persist_path), ingest_lock(persist_path):
        pass

    ingest_lock_holder(persist_path, 30.0)
    with pytest.raises(ConcurrentIngestError), ingest_lock(persist_path):
        pytest.fail("an inner release must not hand the lock to another process")


def test_missing_fcntl_degrades_to_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    persist_path = tmp_path / "chroma"
    monkeypatch.setattr(persist_lock, "fcntl", None)

    with ingest_lock(persist_path), ingest_lock(persist_path):
        pass

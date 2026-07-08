from __future__ import annotations

import threading
import time

import pytest

from localrag.api.jobs import Job, JobRegistry, JobStatus, TooManyPendingJobsError


def _wait_for_terminal(registry: JobRegistry, job_id: str, timeout: float = 5.0) -> Job | None:
    deadline = time.monotonic() + timeout
    job = registry.get(job_id)
    while job is not None and job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        if time.monotonic() > deadline:
            raise AssertionError("job did not finish in time")
        time.sleep(0.01)
        job = registry.get(job_id)
    return job


def test_job_registry_submit_runs_work_in_background_and_reports_done() -> None:
    registry = JobRegistry()
    job_id = registry.submit(lambda: {"value": 42})

    job = _wait_for_terminal(registry, job_id)

    assert job is not None
    assert job.status == JobStatus.DONE
    assert job.result == {"value": 42}


def test_job_registry_marks_failed_on_exception() -> None:
    registry = JobRegistry()

    def boom() -> dict[str, object]:
        raise RuntimeError("nope")

    job_id = registry.submit(boom)
    job = _wait_for_terminal(registry, job_id)

    assert job is not None
    assert job.status == JobStatus.FAILED
    assert job.error == "nope"


def test_job_registry_get_returns_none_for_unknown_job() -> None:
    registry = JobRegistry()
    assert registry.get("nope") is None


def test_job_registry_submit_rejects_past_pending_cap() -> None:
    registry = JobRegistry()
    release = threading.Event()
    job_ids = [registry.submit(release.wait, max_pending=2) for _ in range(2)]

    with pytest.raises(TooManyPendingJobsError):
        registry.submit(lambda: {"value": 1}, max_pending=2)

    release.set()
    for job_id in job_ids:
        _wait_for_terminal(registry, job_id)

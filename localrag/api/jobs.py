"""In-memory background job registry for long-running ingest operations.

Jobs live only for the lifetime of the API process — there is no persistence
across restarts. This matches LocalRAG's single-process, offline-first scope;
a durable job queue backed by an external broker is intentionally out of scope.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class TooManyPendingJobsError(Exception):
    """Raised when a submission would exceed the configured pending/running job cap."""


@dataclass
class Job:
    job_id: str
    status: JobStatus
    created_at: str
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class JobRegistry:
    _jobs: dict[str, Job] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest-job")
    )

    def submit(self, work: Callable[[], dict[str, Any]], *, max_pending: int = 10) -> str:
        job_id = uuid4().hex
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._lock:
            pending = sum(
                1 for j in self._jobs.values() if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
            )
            if pending >= max_pending:
                message = f"{pending} ingest jobs already pending/running (max {max_pending})."
                raise TooManyPendingJobsError(message)
            self._jobs[job_id] = job
        self._executor.submit(self._run, job_id, work)
        return job_id

    def _run(self, job_id: str, work: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            self._jobs[job_id].status = JobStatus.RUNNING
        try:
            result = work()
        except Exception as exc:  # captured for the caller, not re-raised in a background thread
            logger.exception("ingest_job_failed job_id=%s", job_id)
            with self._lock:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].error = str(exc)
            return
        with self._lock:
            self._jobs[job_id].status = JobStatus.DONE
            self._jobs[job_id].result = result

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return Job(
                job_id=job.job_id,
                status=job.status,
                created_at=job.created_at,
                result=job.result,
                error=job.error,
            )

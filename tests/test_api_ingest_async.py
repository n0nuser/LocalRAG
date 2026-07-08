from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from localrag.api.dependencies import get_api_settings, get_ingestion_service, get_job_registry
from localrag.api.jobs import JobRegistry
from localrag.api.main import app
from localrag.ingestion.service import IngestionResult
from localrag.settings import Settings


@dataclass
class StubIngestionService:
    result: IngestionResult

    def ingest_directory(
        self, path: Path, recursive: bool | None = None, embed_model: str | None = None
    ) -> IngestionResult:
        return self.result


def test_ingest_directory_async_returns_job_id_then_reports_done(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()

    settings = Settings(ingest_roots=[str(allowed_root)])
    ingestion = StubIngestionService(
        result=IngestionResult(files_processed=1, total_chunks=4, processed_sources=[])
    )
    registry = JobRegistry()

    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_job_registry] = lambda: registry
    client = TestClient(app)

    submitted = client.post("/ingest/directory/async", json={"path": str(allowed_root)})
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert submitted.json()["status"] == "pending"

    deadline = time.monotonic() + 5.0
    status_body: dict[str, object] = {}
    while time.monotonic() < deadline:
        polled = client.get(f"/ingest/jobs/{job_id}")
        assert polled.status_code == 200
        status_body = polled.json()
        if status_body["status"] == "done":
            break
        time.sleep(0.01)

    assert status_body["status"] == "done"
    assert status_body["result"]["total_chunks"] == 4  # type: ignore[index]

    app.dependency_overrides.clear()


def test_ingest_job_status_unknown_id_returns_404() -> None:
    registry = JobRegistry()
    app.dependency_overrides[get_job_registry] = lambda: registry
    client = TestClient(app)

    response = client.get("/ingest/jobs/does-not-exist")
    assert response.status_code == 404

    app.dependency_overrides.clear()

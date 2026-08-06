from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from localrag.cli import app as app_module
from localrag.cli.commands import ingest as ingest_command
from localrag.ingestion.service import FailedSource, IngestionResult

runner = CliRunner()


class StubService:
    def __init__(self, result: IngestionResult) -> None:
        self.result = result

    def ingest_file(self, **_: Any) -> IngestionResult:
        return self.result

    def ingest_directory(self, **_: Any) -> IngestionResult:
        return self.result


def _run(monkeypatch: Any, result: IngestionResult, tmp_path: Any) -> Any:
    monkeypatch.setattr(ingest_command, "get_ingestion_service", lambda: StubService(result))
    target = tmp_path / "doc.md"
    target.write_text("hello", encoding="utf-8")
    return runner.invoke(app_module.app, ["ingest", str(target)])


def test_ingest_reports_ok_when_everything_succeeded(monkeypatch: Any, tmp_path: Any) -> None:
    result = _run(
        monkeypatch,
        IngestionResult(
            files_processed=2, total_chunks=10, processed_sources=["a", "b"], failed_sources=[]
        ),
        tmp_path,
    )

    assert result.exit_code == 0
    assert "status=ok" in result.stdout
    assert "files_processed=2" in result.stdout


def test_ingest_reports_error_and_exits_non_zero_when_all_files_failed(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A run that ingested nothing must not announce success.

    Reporting status=ok with exit 0 here makes a total failure look identical to
    a clean run in scripts and CI.
    """
    result = _run(
        monkeypatch,
        IngestionResult(
            files_processed=0,
            total_chunks=0,
            processed_sources=[],
            failed_sources=[FailedSource(source="a.epub", error="boom")],
        ),
        tmp_path,
    )

    assert result.exit_code != 0
    assert "status=error" in result.stdout
    assert "failed=1" in result.stdout


def test_ingest_reports_partial_when_some_files_failed(monkeypatch: Any, tmp_path: Any) -> None:
    """Partial success is distinct from both outcomes and must exit non-zero."""
    result = _run(
        monkeypatch,
        IngestionResult(
            files_processed=1,
            total_chunks=5,
            processed_sources=["a"],
            failed_sources=[FailedSource(source="b.epub", error="boom")],
        ),
        tmp_path,
    )

    assert result.exit_code != 0
    assert "status=partial" in result.stdout
    assert "files_processed=1" in result.stdout
    assert "failed=1" in result.stdout

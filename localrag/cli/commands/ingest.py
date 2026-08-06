from __future__ import annotations

import logging
from pathlib import Path

import typer

from localrag.application.container import get_ingestion_service
from localrag.ingestion.service import IngestProgress

logger = logging.getLogger(__name__)


def _echo_progress(event: IngestProgress) -> None:
    """Report one file's outcome to stderr, keeping stdout to the final summary."""
    name = Path(event.source).name
    if event.error is not None:
        detail = f"failed: {event.error}"
    elif event.chunks_added is None:
        detail = "skipped (no chunks)"
    else:
        detail = f"{event.chunks_added} chunks"
    typer.echo(f"[{event.file_index}/{event.file_count}] {name} — {detail}", err=True)


def ingest(path: str, recursive: bool | None = None, *, quiet: bool = False) -> None:
    service = get_ingestion_service()
    target = Path(path)
    logger.info("cli_ingest path=%s recursive=%s is_dir=%s", path, recursive, target.is_dir())

    # Ingest is the slowest operation here; without per-file output a healthy long
    # run is indistinguishable from a hung one.
    on_progress = None if quiet else _echo_progress

    if target.is_dir():
        result = service.ingest_directory(path=target, recursive=recursive, on_progress=on_progress)
    else:
        result = service.ingest_file(path=target, on_progress=on_progress)

    failed_count = len(result.failed_sources)
    logger.info(
        "cli_ingest_done files=%s chunks=%s failed=%s",
        result.files_processed,
        result.total_chunks,
        failed_count,
    )

    # A run that ingested nothing must not announce success: scripts and CI read
    # the exit code, and reporting ok made a total failure look like a clean run.
    if failed_count == 0:
        status = "ok"
    elif result.files_processed > 0:
        status = "partial"
    else:
        status = "error"

    summary = (
        f"status={status} files_processed={result.files_processed} "
        f"total_chunks={result.total_chunks}"
    )
    if failed_count:
        summary = f"{summary} failed={failed_count}"
    typer.echo(summary)

    if failed_count:
        for failure in result.failed_sources:
            typer.echo(f"failed: {failure.source} — {failure.error}", err=True)
        raise typer.Exit(code=1)

from __future__ import annotations

from pathlib import Path

import typer

from hermit.api.dependencies import get_ingestion_service


def ingest(path: str, recursive: bool | None = None) -> None:
    service = get_ingestion_service()
    target = Path(path)

    if target.is_dir():
        result = service.ingest_directory(path=target, recursive=recursive)
    else:
        result = service.ingest_file(path=target)

    typer.echo(
        f"status=ok files_processed={result.files_processed} total_chunks={result.total_chunks}"
    )

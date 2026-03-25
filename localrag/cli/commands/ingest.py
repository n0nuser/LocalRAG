from __future__ import annotations

import logging
from pathlib import Path

import typer

from localrag.api.dependencies import get_ingestion_service

logger = logging.getLogger(__name__)


def ingest(path: str, recursive: bool | None = None) -> None:
    service = get_ingestion_service()
    target = Path(path)
    logger.info("cli_ingest path=%s recursive=%s is_dir=%s", path, recursive, target.is_dir())

    if target.is_dir():
        result = service.ingest_directory(path=target, recursive=recursive)
    else:
        result = service.ingest_file(path=target)

    logger.info(
        "cli_ingest_done files=%s chunks=%s",
        result.files_processed,
        result.total_chunks,
    )
    typer.echo(
        f"status=ok files_processed={result.files_processed} total_chunks={result.total_chunks}"
    )

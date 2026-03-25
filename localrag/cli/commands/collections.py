from __future__ import annotations

import json
import logging

import typer

from localrag.api.dependencies import get_ingestion_service, get_vector_store

logger = logging.getLogger(__name__)
app = typer.Typer(help="Collection operations")


@app.command("list")
def list_collections() -> None:
    store = get_vector_store()
    rows = store.list_collections()
    logger.debug("cli_collections_list count=%s", len(rows))
    typer.echo(json.dumps(rows, indent=2))


@app.command("delete")
def delete_collection(name: str) -> None:
    logger.warning("cli_collection_delete name=%s", name)
    store = get_vector_store()
    store.delete_collection(name)
    typer.echo("status=ok")


@app.command("rebuild")
def rebuild_collection(embed_model: str | None = typer.Option(None, "--embed-model")) -> None:
    service = get_ingestion_service()
    logger.info("cli_collection_rebuild embed_model=%s", embed_model)
    result = service.rebuild_collection(embed_model=embed_model)
    typer.echo(
        f"status=ok files_processed={result.files_processed} "
        f"total_chunks={result.total_chunks} "
        f"missing_sources={len(result.missing_sources)}"
    )
    if result.missing_sources:
        for src in result.missing_sources:
            typer.echo(f"missing_removed={src}", err=True)

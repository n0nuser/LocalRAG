from __future__ import annotations

import json
import logging

import typer

from localrag.api.dependencies import get_vector_store

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

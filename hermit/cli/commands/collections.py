from __future__ import annotations

import json

import typer

from hermit.api.dependencies import get_vector_store

app = typer.Typer(help="Collection operations")


@app.command("list")
def list_collections() -> None:
    store = get_vector_store()
    typer.echo(json.dumps(store.list_collections(), indent=2))


@app.command("delete")
def delete_collection(name: str) -> None:
    store = get_vector_store()
    store.delete_collection(name)
    typer.echo("status=ok")

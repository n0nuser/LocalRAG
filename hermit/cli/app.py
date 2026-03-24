from __future__ import annotations

import typer

from hermit.cli.commands.collections import app as collections_app
from hermit.cli.commands.config import show_config
from hermit.cli.commands.ingest import ingest
from hermit.cli.commands.query import query
from hermit.cli.commands.setup import setup

app = typer.Typer(help="Hermit CLI")

app.command()(ingest)
app.command()(query)
app.command("setup")(setup)
app.command("config-show")(show_config)
app.add_typer(collections_app, name="collections")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

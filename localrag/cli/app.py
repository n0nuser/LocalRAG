from __future__ import annotations

import typer

from localrag.cli.commands.collections import app as collections_app
from localrag.cli.commands.config import show_config
from localrag.cli.commands.ingest import ingest
from localrag.cli.commands.query import query
from localrag.cli.commands.setup import setup
from localrag.logging_config import configure_logging
from localrag.settings import get_settings

app = typer.Typer(help="LocalRAG CLI")

app.command()(ingest)
app.command()(query)
app.command("setup")(setup)
app.command("config-show")(show_config)
app.add_typer(collections_app, name="collections")


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()

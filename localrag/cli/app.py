from __future__ import annotations

from typing import Annotated

import typer
import yaml

from localrag.cli.commands.benchmark import benchmark
from localrag.cli.commands.collections import app as collections_app
from localrag.cli.commands.config import show_config
from localrag.cli.commands.eval import eval_suite
from localrag.cli.commands.eval_compare import eval_compare
from localrag.cli.commands.ingest import ingest
from localrag.cli.commands.inspect import inspect
from localrag.cli.commands.leaderboard import leaderboard
from localrag.cli.commands.query import query
from localrag.cli.commands.report import report
from localrag.cli.commands.setup import setup
from localrag.logging_config import configure_logging
from localrag.settings import ConfigError, get_settings, load_settings, set_current_settings

app = typer.Typer(help="LocalRAG CLI")


@app.callback()
def configure(
    config: Annotated[
        str | None,
        typer.Option("--config", help="YAML configuration file loaded before services start."),
    ] = None,
    overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            help="Explicit configuration override as FIELD=VALUE; may be repeated.",
        ),
    ] = None,
) -> None:
    values: dict[str, object] = {}
    for override in overrides or []:
        if "=" not in override:
            raise typer.BadParameter("must use FIELD=VALUE", param_hint="--set")
        field, raw_value = override.split("=", 1)
        if not field:
            raise typer.BadParameter("field name cannot be empty", param_hint="--set")
        values[field] = yaml.safe_load(raw_value)
    try:
        set_current_settings(load_settings(config, values))
    except ConfigError as exc:
        raise typer.BadParameter(str(exc), param_hint="--config/--set") from exc


app.command()(ingest)
app.command()(query)
app.command("setup")(setup)
app.command("config-show")(show_config)
app.command("eval")(eval_suite)
app.command("benchmark")(benchmark)
app.command("inspect")(inspect)
app.command("eval-compare")(eval_compare)
app.command("report")(report)
app.command("leaderboard")(leaderboard)
app.add_typer(collections_app, name="collections")


def main() -> None:
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()

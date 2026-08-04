from __future__ import annotations

import json

import typer

from localrag.settings import get_settings


def show_config() -> None:
    settings = get_settings()
    typer.echo(json.dumps(settings.resolved_snapshot(), indent=2, sort_keys=True))

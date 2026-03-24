from __future__ import annotations

import json

import typer

from hermit.config import get_settings


def show_config() -> None:
    settings = get_settings()
    typer.echo(json.dumps(settings.model_dump(), indent=2))

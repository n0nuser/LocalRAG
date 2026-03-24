from __future__ import annotations

import json

import typer

from hermit.api.dependencies import get_engine


def query(question: str, model: str | None = None, n_results: int | None = None) -> None:
    engine = get_engine()
    sources: list[dict[str, object]] = []

    for event in engine.stream_answer(question=question, model=model, n_results=n_results):
        if event["type"] == "token":
            typer.echo(str(event["token"]), nl=False)
        if event["type"] == "final":
            sources = list(event["sources"])

    typer.echo("")
    typer.echo(f"sources={json.dumps(sources)}")

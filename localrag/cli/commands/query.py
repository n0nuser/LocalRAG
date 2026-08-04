from __future__ import annotations

import json
import logging

import typer

from localrag.api.dependencies import get_engine

logger = logging.getLogger(__name__)


def query(question: str, model: str | None = None, n_results: int | None = None) -> None:
    engine = get_engine()
    logger.info(
        "cli_query question_chars=%s model=%s n_results=%s",
        len(question),
        model,
        n_results,
    )
    sources: list[dict[str, object]] = []
    trace: dict[str, object] | None = None

    for event in engine.stream_answer(question=question, model=model, n_results=n_results):
        if event["type"] == "token":
            typer.echo(str(event["token"]), nl=False)
        if event["type"] == "final":
            sources = list(event["sources"])
            trace = event.get("trace") if isinstance(event.get("trace"), dict) else None

    typer.echo("")
    logger.info("cli_query_done source_count=%s", len(sources))
    typer.echo(f"sources={json.dumps(sources)}")
    if trace is not None:
        typer.echo(f"trace={json.dumps(trace, sort_keys=True)}")

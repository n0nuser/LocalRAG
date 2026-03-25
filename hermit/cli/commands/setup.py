from __future__ import annotations

import logging

import httpx
import typer

from hermit.ollama.schemas import (
    OllamaPullRequest,
    OllamaPullStreamChunk,
    OllamaTagsResponse,
    parse_ollama_json,
    parse_ollama_json_line,
)
from hermit.settings import get_settings

logger = logging.getLogger(__name__)


def setup() -> None:
    settings = get_settings()
    logger.info("cli_setup_start ollama_url=%s", settings.ollama_base_url)
    typer.echo("Checking Ollama...")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            logger.debug("cli_setup_tags_ok")
            tags = parse_ollama_json(response.json(), OllamaTagsResponse)
            existing_models = {m.name.split(":")[0] for m in tags.models}

            for model in [settings.ollama_embed_model, settings.ollama_llm_model]:
                if model in existing_models:
                    logger.info("cli_setup_model_present model=%s", model)
                    typer.echo(f"Model already available: {model}")
                    continue

                logger.warning("cli_setup_pull_start model=%s", model)
                typer.echo(f"Pulling model: {model}")
                pull_body = OllamaPullRequest(model=model, stream=True)
                with client.stream(
                    "POST",
                    f"{settings.ollama_base_url}/api/pull",
                    json=pull_body.model_dump(mode="json", exclude_none=True),
                ) as pull_response:
                    pull_response.raise_for_status()
                    for line in pull_response.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = parse_ollama_json_line(line, OllamaPullStreamChunk)
                        except ValueError:
                            logger.debug("cli_setup_pull_unparsed_line chars=%s", len(line))
                            continue
                        if chunk.status:
                            typer.echo(chunk.status)

    except httpx.HTTPError as exc:
        logger.error("cli_setup_ollama_unreachable url=%s error=%s", settings.ollama_base_url, exc)
        raise

    logger.info("cli_setup_complete")
    typer.echo("Setup complete.")

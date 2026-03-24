from __future__ import annotations

import json

import httpx
import typer

from hermit.config import get_settings


def setup() -> None:
    settings = get_settings()
    typer.echo("Checking Ollama...")

    with httpx.Client(timeout=30.0) as client:
        response = client.get(f"{settings.ollama_base_url}/api/tags")
        response.raise_for_status()
        existing_models = {
            model["name"].split(":")[0] for model in response.json().get("models", [])
        }

        for model in [settings.ollama_embed_model, settings.ollama_llm_model]:
            if model in existing_models:
                typer.echo(f"Model already available: {model}")
                continue

            typer.echo(f"Pulling model: {model}")
            with client.stream(
                "POST",
                f"{settings.ollama_base_url}/api/pull",
                json={"name": model, "stream": True},
            ) as pull_response:
                pull_response.raise_for_status()
                for line in pull_response.iter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    status = payload.get("status")
                    if isinstance(status, str):
                        typer.echo(status)

    typer.echo("Setup complete.")

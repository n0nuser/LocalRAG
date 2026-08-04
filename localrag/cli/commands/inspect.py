"""Read-only collection diagnostics for local development and scripts."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import typer

from localrag.settings import get_settings
from localrag.storage.vector_store import VectorStore

INSPECT_SCHEMA_VERSION = 1
_SENSITIVE = re.compile(r"(?:api[_-]?key|password|secret|token)", re.IGNORECASE)


class CollectionNotFoundError(Exception):
    """The requested collection is not present in the local store."""


def open_collection(name: str) -> Any:
    """Open a configured local collection without creating it."""
    settings = get_settings()
    try:
        return VectorStore.open(settings.chroma_persist_path, name).collection
    except Exception as exc:
        if "does not exist" in str(exc).lower() or "not found" in str(exc).lower():
            message = f"collection {name!r} does not exist"
            raise CollectionNotFoundError(message) from exc
        raise


def _safe_text(value: Any, limit: int) -> str:
    text = str(value).replace("\x00", "�")
    text = "".join(character if character.isprintable() else " " for character in text)
    if len(text) <= limit and len(text.encode("utf-8")) <= limit:
        return text
    marker = "…[truncated]"
    if limit < len(marker):
        return marker[:limit]
    prefix = text[: max(0, limit - len(marker))]
    while len(prefix.encode("utf-8")) + len(marker.encode("utf-8")) > limit:
        prefix = prefix[:-1]
    return prefix + marker


def _safe_metadata(metadata: Any, limit: int) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {"_malformed": {"type": type(metadata).__name__}}
    result: dict[str, Any] = {}
    for key in sorted(metadata, key=str):
        key_text = str(key)
        if _SENSITIVE.search(key_text):
            result[key_text] = "[redacted]"
        else:
            result[key_text] = _safe_text(metadata[key], limit)
    return result


def _metadata_summary(collection: Any, limit: int) -> dict[str, Any]:
    raw = collection.get(include=["metadatas"], limit=1000)
    values = raw.get("metadatas") or []
    key_types: dict[str, Counter[str]] = {}
    key_values: dict[str, set[str]] = {}
    for metadata in values:
        if not isinstance(metadata, dict):
            continue
        for key, value in metadata.items():
            key_text = str(key)
            if _SENSITIVE.search(key_text):
                continue
            key_types.setdefault(key_text, Counter())[type(value).__name__] += 1
            key_values.setdefault(key_text, set()).add(_safe_text(value, limit))
    return {
        key: {
            "count": sum(counts.values()),
            "types": dict(sorted(counts.items())),
            "values": sorted(key_values[key])[:5],
        }
        for key, counts in sorted(key_types.items())
    }


def inspect_collection(collection_name: str, sample_count: int, max_chars: int) -> dict[str, Any]:
    collection = open_collection(collection_name)
    raw = collection.get(include=["ids", "documents", "metadatas"], limit=sample_count)
    rows = sorted(
        zip(
            raw.get("ids") or [],
            raw.get("documents") or [],
            raw.get("metadatas") or [],
            strict=False,
        ),
        key=lambda row: str(row[0]),
    )
    samples = [
        {
            "id": _safe_text(chunk_id, max_chars),
            "document": _safe_text(document, max_chars),
            "metadata": _safe_metadata(metadata, max_chars),
        }
        for chunk_id, document, metadata in rows[:sample_count]
    ]
    return {
        "schema_version": INSPECT_SCHEMA_VERSION,
        "collection": collection_name,
        "vector_count": collection.count(),
        "document_count": collection.count(),
        "collection_metadata": _safe_metadata(getattr(collection, "metadata", {}), max_chars),
        "metadata_summary": _metadata_summary(collection, max_chars),
        "samples": samples,
    }


def inspect(
    collection: str = typer.Option("localrag", help="Collection to inspect."),
    sample_count: int = typer.Option(5, min=0, max=100, help="Maximum deterministic sample count."),
    max_chars: int = typer.Option(500, min=16, max=10_000, help="Hard limit per displayed value."),
    output_format: str = typer.Option("table", "--format", help="Output format: table or json."),
) -> None:
    """Inspect local Chroma data without invoking the API, LLM, or network."""
    if output_format not in {"table", "json"}:
        raise typer.BadParameter("must be table or json", param_hint="--format")
    try:
        payload = inspect_collection(collection, sample_count, max_chars)
    except CollectionNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"inspect failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if output_format == "json":
        typer.echo(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        return
    typer.echo(f"Collection: {payload['collection']}")
    typer.echo(f"Vectors: {payload['vector_count']}")
    typer.echo(f"Documents: {payload['document_count']}")
    typer.echo("Metadata keys: " + (", ".join(payload["metadata_summary"]) or "none"))
    typer.echo("Samples:")
    for sample in payload["samples"]:
        typer.echo(f"- {sample['id']}: {sample['document']}")

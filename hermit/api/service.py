from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx

from hermit.api.exceptions import IngestApiError
from hermit.api.repository import ChromaCollectionRepository
from hermit.api.schemas import (
    CollectionDeleteResponse,
    CollectionListResponse,
    HealthResponse,
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestFileResponse,
    QueryRequest,
)
from hermit.ingestion.service import IngestionService
from hermit.ollama.schemas import OllamaTagsResponse, parse_ollama_json
from hermit.rag.engine import RAGEngine
from hermit.settings import Settings, is_path_allowed

logger = logging.getLogger(__name__)


def path_from_ingest_request(raw: str) -> Path:
    # Clients often copy URL-encoded paths (%20); the OS expects decoded characters.
    return Path(unquote(raw.strip()))


def check_health(settings: Settings, collection_repo: ChromaCollectionRepository) -> HealthResponse:
    ollama_ok = False
    with httpx.Client(timeout=5.0) as client:
        try:
            response = client.get(f"{settings.ollama_base_url}/api/tags")
            response.raise_for_status()
            parse_ollama_json(response.json(), OllamaTagsResponse)
            ollama_ok = True
        except httpx.HTTPError:
            ollama_ok = False
            logger.warning("health_ollama_unreachable url=%s", settings.ollama_base_url)
        except ValueError as exc:
            ollama_ok = False
            logger.warning(
                "health_ollama_tags_invalid url=%s error=%s", settings.ollama_base_url, exc
            )

    logger.debug("health_check ollama_ok=%s", ollama_ok)

    return HealthResponse(
        status="ok",
        ollama_ok=ollama_ok,
        chroma_path=settings.chroma_persist_path,
        collections=collection_repo.list_collection_names(),
    )


def list_collections_response(
    collection_repo: ChromaCollectionRepository,
) -> CollectionListResponse:
    names = collection_repo.list_collection_names()
    logger.debug("collections_list count=%s", len(names))
    return CollectionListResponse(collections=names)


def delete_collection_response(
    collection_repo: ChromaCollectionRepository, name: str
) -> CollectionDeleteResponse:
    logger.warning("collection_delete name=%s", name)
    collection_repo.delete_collection(name)
    return CollectionDeleteResponse(status="ok")


def ingest_file(
    request: IngestFileRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> IngestFileResponse:
    path = path_from_ingest_request(request.path).resolve()
    if not path.is_file():
        logger.warning("ingest_file_rejected not_a_file path=%s", path)
        raise IngestApiError(HTTPStatus.BAD_REQUEST, "Path must be an existing file.")
    if not is_path_allowed(path, settings.ingest_roots):
        logger.warning("ingest_file_rejected outside_roots path=%s", path)
        raise IngestApiError(
            HTTPStatus.FORBIDDEN,
            "Path is not under configured ingest roots.",
        )
    logger.info("ingest_file_start path=%s", path)
    result = ingestion_service.ingest_file(path)
    logger.info(
        "ingest_file_done path=%s chunks=%s",
        path,
        result.total_chunks,
    )
    source = result.processed_sources[0] if result.processed_sources else request.path
    return IngestFileResponse(
        status="ok",
        chunks_added=result.total_chunks,
        source=str(source),
    )


def ingest_directory(
    request: IngestDirectoryRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> IngestDirectoryResponse:
    path = path_from_ingest_request(request.path).resolve()
    if not path.is_dir():
        logger.warning("ingest_directory_rejected not_a_dir path=%s", path)
        raise IngestApiError(HTTPStatus.BAD_REQUEST, "Path must be an existing directory.")
    if not is_path_allowed(path, settings.ingest_roots):
        logger.warning("ingest_directory_rejected outside_roots path=%s", path)
        raise IngestApiError(
            HTTPStatus.FORBIDDEN,
            "Path is not under configured ingest roots.",
        )
    logger.info(
        "ingest_directory_start path=%s recursive=%s",
        path,
        request.recursive,
    )
    result = ingestion_service.ingest_directory(path, recursive=request.recursive)
    logger.info(
        "ingest_directory_done path=%s files=%s chunks=%s",
        path,
        result.files_processed,
        result.total_chunks,
    )
    return IngestDirectoryResponse(
        status="ok",
        files_processed=result.files_processed,
        total_chunks=result.total_chunks,
    )


def iter_query_sse_events(request: QueryRequest, engine: RAGEngine) -> Iterator[dict[str, Any]]:
    logger.info(
        "query_start model=%s n_results=%s question_chars=%s",
        request.model,
        request.n_results,
        len(request.question),
    )
    for event in engine.stream_answer(
        question=request.question,
        model=request.model,
        n_results=request.n_results,
    ):
        if event["type"] == "token":
            yield {"event": "token", "data": str(event["token"])}
        if event["type"] == "final":
            payload = {"sources": event["sources"]}
            yield {"event": "final", "data": json.dumps(payload)}

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, BinaryIO

from localrag.api import schemas
from localrag.application import service as application_service
from localrag.application.dto import (
    IngestDirectoryRequest,
    IngestFileRequest,
    QueryRequest,
    RebuildCollectionRequest,
)
from localrag.application.jobs import JobRegistry
from localrag.application.repository import ChromaCollectionRepository
from localrag.ingestion.service import IngestionService
from localrag.rag.engine import RAGEngine
from localrag.rag.query_cache import QueryCache
from localrag.settings import Settings


def check_readiness(
    settings: Settings, collection_repo: ChromaCollectionRepository
) -> schemas.ReadinessResponse:
    result = application_service.check_readiness(settings, collection_repo)
    return schemas.ReadinessResponse(status=result.status)


def list_collections_response(
    collection_repo: ChromaCollectionRepository,
) -> schemas.CollectionListResponse:
    result = application_service.list_collections_response(collection_repo)
    return schemas.CollectionListResponse(collections=result.collections)


def delete_collection_response(
    collection_repo: ChromaCollectionRepository, name: str
) -> schemas.CollectionDeleteResponse:
    result = application_service.delete_collection_response(collection_repo, name)
    return schemas.CollectionDeleteResponse(status=result.status)


def rebuild_collection_response(
    request: schemas.RebuildCollectionRequest,
    ingestion_service: IngestionService,
) -> schemas.RebuildCollectionResponse:
    result = application_service.rebuild_collection_response(
        RebuildCollectionRequest(embed_model=request.embed_model), ingestion_service
    )
    return schemas.RebuildCollectionResponse(
        status=result.status,
        files_processed=result.files_processed,
        total_chunks=result.total_chunks,
        missing_sources=result.missing_sources,
        failed_sources=[
            schemas.FailedSourceRef(**vars(source)) for source in result.failed_sources
        ],
        skipped_unchanged_sources=result.skipped_unchanged_sources,
    )


def ingest_file(
    request: schemas.IngestFileRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> schemas.IngestFileResponse:
    result = application_service.ingest_file(
        IngestFileRequest(path=request.path, embed_model=request.embed_model),
        settings,
        ingestion_service,
    )
    return schemas.IngestFileResponse(**vars(result))


def ingest_directory(
    request: schemas.IngestDirectoryRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> schemas.IngestDirectoryResponse:
    result = application_service.ingest_directory(
        IngestDirectoryRequest(
            path=request.path,
            recursive=request.recursive,
            embed_model=request.embed_model,
        ),
        settings,
        ingestion_service,
    )
    return schemas.IngestDirectoryResponse(
        status=result.status,
        files_processed=result.files_processed,
        total_chunks=result.total_chunks,
        failed_sources=[
            schemas.FailedSourceRef(**vars(source)) for source in result.failed_sources
        ],
    )


def ingest_directory_async(
    request: schemas.IngestDirectoryRequest,
    settings: Settings,
    ingestion_service: IngestionService,
    job_registry: JobRegistry,
) -> schemas.IngestJobResponse:
    result = application_service.ingest_directory_async(
        IngestDirectoryRequest(
            path=request.path,
            recursive=request.recursive,
            embed_model=request.embed_model,
        ),
        settings,
        ingestion_service,
        job_registry,
    )
    return schemas.IngestJobResponse(job_id=result.job_id, status=result.status)


def get_ingest_job(job_id: str, job_registry: JobRegistry) -> schemas.IngestJobStatusResponse:
    result = application_service.get_ingest_job(job_id, job_registry)
    return schemas.IngestJobStatusResponse(
        job_id=result.job_id,
        status=result.status,
        result=result.result,
        error=result.error,
    )


def ingest_upload(
    file_name: str,
    file_obj: BinaryIO,
    embed_model: str | None,
    settings: Settings,
    ingestion_service: IngestionService,
) -> schemas.IngestFileResponse:
    result = application_service.ingest_upload(
        file_name, file_obj, embed_model, settings, ingestion_service
    )
    return schemas.IngestFileResponse(**vars(result))


def query_json(
    request: schemas.QueryRequest, engine: RAGEngine, query_cache: QueryCache | None = None
) -> schemas.QueryResponse:
    result = application_service.query_json(
        QueryRequest(
            question=request.question,
            model=request.model,
            n_results=request.n_results,
            metadata_filter=request.metadata_filter,
        ),
        engine,
        query_cache,
    )
    return schemas.QueryResponse(
        answer=result.answer,
        sources=[schemas.SourceRef(**vars(source)) for source in result.sources],
        latency_ms=result.latency_ms,
        model=result.model,
        provider=result.provider,
        low_confidence=result.low_confidence,
        trace=result.trace,
    )


def get_query_contexts(request: schemas.QueryRequest, engine: RAGEngine) -> list[dict[str, Any]]:
    return application_service.get_query_contexts(
        QueryRequest(
            question=request.question,
            model=request.model,
            n_results=request.n_results,
            metadata_filter=request.metadata_filter,
        ),
        engine,
    )


def iter_query_sse_events(
    request: schemas.QueryRequest,
    engine: RAGEngine,
    contexts: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    return application_service.iter_query_sse_events(
        QueryRequest(
            question=request.question,
            model=request.model,
            n_results=request.n_results,
            metadata_filter=request.metadata_filter,
        ),
        engine,
        contexts,
    )

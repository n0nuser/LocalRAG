from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO, cast
from urllib.parse import unquote
from uuid import uuid4

import httpx

from localrag import metrics as app_metrics
from localrag.application.dto import (
    CollectionDeleteResponse,
    CollectionListResponse,
    FailedSourceRef,
    IngestDirectoryRequest,
    IngestDirectoryResponse,
    IngestFileRequest,
    IngestFileResponse,
    IngestJobResponse,
    IngestJobStatusResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
    RebuildCollectionRequest,
    RebuildCollectionResponse,
    SourceRef,
)
from localrag.application.errors import ApplicationErrorKind, IngestError, QueryError
from localrag.application.jobs import JobRegistry, TooManyPendingJobsError
from localrag.application.repository import ChromaCollectionRepository
from localrag.audit import write_audit_record
from localrag.ingestion.loader import SUPPORTED_EXTENSIONS
from localrag.ingestion.service import IngestionResult, IngestionService
from localrag.logging_config import request_id_ctx
from localrag.observability.tracing import SpanName, span
from localrag.ollama.schemas import OllamaTagsResponse, parse_ollama_json
from localrag.rag.engine import RAGEngine
from localrag.rag.exceptions import RetrievalError, RetrievalFailureKind
from localrag.rag.query_cache import QueryCache, make_cache_key
from localrag.rag.retriever import Retriever
from localrag.settings import Settings, is_path_allowed

logger = logging.getLogger(__name__)


def path_from_ingest_request(raw: str) -> Path:
    # Clients often copy URL-encoded paths (%20); the OS expects decoded characters.
    return Path(unquote(raw.strip()))


def check_readiness(
    settings: Settings, collection_repo: ChromaCollectionRepository
) -> ReadinessResponse:
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

    try:
        collection_repo.list_collection_names()
        chroma_ok = True
    except Exception:
        chroma_ok = False
        logger.exception("health_chroma_unreachable")

    ready = ollama_ok and chroma_ok
    logger.debug("readiness_check ollama_ok=%s chroma_ok=%s", ollama_ok, chroma_ok)
    return ReadinessResponse(status="ok" if ready else "unavailable")


def list_collections_response(
    collection_repo: ChromaCollectionRepository,
) -> CollectionListResponse:
    names = collection_repo.list_collection_names()
    logger.debug("collections_list count=%s", len(names))
    return CollectionListResponse(collections=names)


def delete_collection_response(
    collection_repo: ChromaCollectionRepository, name: str
) -> CollectionDeleteResponse:
    if name not in collection_repo.list_collection_names():
        raise IngestError(ApplicationErrorKind.NOT_FOUND, f"Collection '{name}' not found.")
    logger.warning("collection_delete name=%s", name)
    collection_repo.delete_collection(name)
    return CollectionDeleteResponse(status="ok")


def rebuild_collection_response(
    request: RebuildCollectionRequest,
    ingestion_service: IngestionService,
) -> RebuildCollectionResponse:
    logger.info("collection_rebuild_start embed_model=%s", request.embed_model)
    result = ingestion_service.rebuild_collection(embed_model=request.embed_model)
    logger.info(
        "collection_rebuild_done files=%s chunks=%s missing=%s failed=%s",
        result.files_processed,
        result.total_chunks,
        len(result.missing_sources),
        len(result.failed_sources),
    )
    return RebuildCollectionResponse(
        status="ok",
        files_processed=result.files_processed,
        total_chunks=result.total_chunks,
        missing_sources=result.missing_sources,
        failed_sources=[
            FailedSourceRef(source=f.source, error=f.error) for f in result.failed_sources
        ],
        skipped_unchanged_sources=result.skipped_unchanged_sources,
    )


def ingest_file(
    request: IngestFileRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> IngestFileResponse:
    path = path_from_ingest_request(request.path).resolve()
    if not path.is_file():
        logger.warning("ingest_file_rejected not_a_file path=%s", path)
        raise IngestError(ApplicationErrorKind.BAD_REQUEST, "Path must be an existing file.")
    if not is_path_allowed(path, settings.ingest_roots):
        logger.warning("ingest_file_rejected outside_roots path=%s", path)
        raise IngestError(
            ApplicationErrorKind.FORBIDDEN,
            "Path is not under configured ingest roots.",
        )
    logger.info("ingest_file_start path=%s", path)
    result = ingestion_service.ingest_file(path, embed_model=request.embed_model)
    _raise_if_failed(result)
    logger.info(
        "ingest_file_done path=%s chunks=%s",
        path,
        result.total_chunks,
    )
    app_metrics.ingested_documents_total.inc(1)
    source = result.processed_sources[0] if result.processed_sources else request.path
    return IngestFileResponse(
        status="ok",
        chunks_added=result.total_chunks,
        source=str(source),
    )


def _raise_if_failed(result: IngestionResult) -> None:
    """Single-file endpoints retry once internally, then surface a hard failure as an error.

    Batch endpoints (`ingest_directory`, `rebuild_collection`) instead report
    `failed_sources` in their JSON response, since partial success there is expected.
    """
    if not result.failed_sources:
        return
    failure = result.failed_sources[0]
    raise IngestError(
        ApplicationErrorKind.BAD_GATEWAY,
        f"Ingest failed after retry: {failure.error}",
    )


def ingest_directory(
    request: IngestDirectoryRequest,
    settings: Settings,
    ingestion_service: IngestionService,
) -> IngestDirectoryResponse:
    path = path_from_ingest_request(request.path).resolve()
    if not path.is_dir():
        logger.warning("ingest_directory_rejected not_a_dir path=%s", path)
        raise IngestError(ApplicationErrorKind.BAD_REQUEST, "Path must be an existing directory.")
    if not is_path_allowed(path, settings.ingest_roots):
        logger.warning("ingest_directory_rejected outside_roots path=%s", path)
        raise IngestError(
            ApplicationErrorKind.FORBIDDEN,
            "Path is not under configured ingest roots.",
        )
    logger.info(
        "ingest_directory_start path=%s recursive=%s",
        path,
        request.recursive,
    )
    result = ingestion_service.ingest_directory(
        path,
        recursive=request.recursive,
        embed_model=request.embed_model,
    )
    logger.info(
        "ingest_directory_done path=%s files=%s chunks=%s failed=%s",
        path,
        result.files_processed,
        result.total_chunks,
        len(result.failed_sources),
    )
    app_metrics.ingested_documents_total.inc(result.files_processed)
    return IngestDirectoryResponse(
        status="ok",
        files_processed=result.files_processed,
        total_chunks=result.total_chunks,
        failed_sources=[
            FailedSourceRef(source=f.source, error=f.error) for f in result.failed_sources
        ],
    )


def ingest_directory_async(
    request: IngestDirectoryRequest,
    settings: Settings,
    ingestion_service: IngestionService,
    job_registry: JobRegistry,
) -> IngestJobResponse:
    path = path_from_ingest_request(request.path).resolve()
    if not path.is_dir():
        raise IngestError(ApplicationErrorKind.BAD_REQUEST, "Path must be an existing directory.")
    if not is_path_allowed(path, settings.ingest_roots):
        raise IngestError(
            ApplicationErrorKind.FORBIDDEN, "Path is not under configured ingest roots."
        )

    def work() -> dict[str, Any]:
        result = ingestion_service.ingest_directory(
            path, recursive=request.recursive, embed_model=request.embed_model
        )
        app_metrics.ingested_documents_total.inc(result.files_processed)
        return {
            "status": "ok",
            "files_processed": result.files_processed,
            "total_chunks": result.total_chunks,
            "failed_sources": [
                {"source": f.source, "error": f.error} for f in result.failed_sources
            ],
        }

    try:
        job_id = job_registry.submit(work, max_pending=settings.max_pending_ingest_jobs)
    except TooManyPendingJobsError as exc:
        raise IngestError(ApplicationErrorKind.TOO_MANY_REQUESTS, str(exc)) from exc
    logger.info("ingest_directory_async_submitted job_id=%s path=%s", job_id, path)
    return IngestJobResponse(job_id=job_id, status="pending")


def get_ingest_job(job_id: str, job_registry: JobRegistry) -> IngestJobStatusResponse:
    job = job_registry.get(job_id)
    if job is None:
        raise IngestError(ApplicationErrorKind.NOT_FOUND, f"Unknown job id '{job_id}'.")
    return IngestJobStatusResponse(
        job_id=job.job_id, status=job.status.value, result=job.result, error=job.error
    )


def ingest_upload(
    file_name: str,
    file_obj: BinaryIO,
    embed_model: str | None,
    settings: Settings,
    ingestion_service: IngestionService,
) -> IngestFileResponse:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        logger.warning("ingest_upload_rejected unsupported_extension name=%s", file_name)
        raise IngestError(
            ApplicationErrorKind.UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file extension '{extension}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    dest_dir = Path(settings.upload_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_uploads(dest_dir, settings)
    temp_path = dest_dir / f".uploading-{uuid4().hex}"
    dest_path: Path | None = None
    try:
        bytes_written, digest = _stream_upload_to_disk(
            file_obj, temp_path, settings.upload_max_bytes
        )
        if bytes_written > settings.upload_quota_bytes:
            app_metrics.upload_quota_rejections_total.inc()
            raise IngestError(
                ApplicationErrorKind.INSUFFICIENT_STORAGE,
                "Upload exceeds the configured upload quota.",
            )
        dest_path = dest_dir / f"{digest}{extension}"
        if dest_path.exists():
            temp_path.unlink(missing_ok=True)
            app_metrics.upload_cleanup_total.labels(reason="deduplicated").inc()
        else:
            temp_path.replace(dest_path)
        _cleanup_uploads(dest_dir, settings, protected=dest_path)
        if not dest_path.exists():
            app_metrics.upload_quota_rejections_total.inc()
            raise IngestError(
                ApplicationErrorKind.INSUFFICIENT_STORAGE, "Upload quota is exhausted."
            )
        logger.info("ingest_upload_saved path=%s bytes=%s", dest_path, bytes_written)
        result = ingestion_service.ingest_file(dest_path, embed_model=embed_model)
        _raise_if_failed(result)
        logger.info("ingest_upload_done path=%s chunks=%s", dest_path, result.total_chunks)
        app_metrics.ingested_documents_total.inc(1)
        source = result.processed_sources[0] if result.processed_sources else dest_path
        return IngestFileResponse(status="ok", chunks_added=result.total_chunks, source=str(source))
    finally:
        temp_path.unlink(missing_ok=True)
        if dest_path is not None and settings.upload_retention_seconds <= 0:
            dest_path.unlink(missing_ok=True)
            app_metrics.upload_cleanup_total.labels(reason="retention").inc()


def _stream_upload_to_disk(file_obj: BinaryIO, dest_path: Path, max_bytes: int) -> tuple[int, str]:
    chunk_size = 1024 * 1024
    bytes_written = 0
    digest = hashlib.sha256()
    completed = False
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise IngestError(
                        ApplicationErrorKind.REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds the {max_bytes}-byte upload limit.",
                    )
                out.write(chunk)
                digest.update(chunk)
        completed = True
        return bytes_written, digest.hexdigest()
    finally:
        if not completed:
            dest_path.unlink(missing_ok=True)


def _cleanup_uploads(directory: Path, settings: Settings, protected: Path | None = None) -> None:
    now = time.time()
    files = []
    for path in directory.iterdir():
        if path.is_file() and not path.name.startswith(".uploading-"):
            try:
                stat = path.stat()
            except OSError:
                continue
            if path != protected and (
                settings.upload_retention_seconds <= 0
                or now - stat.st_mtime > settings.upload_retention_seconds
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("upload_cleanup_failed path=%s", path)
                    continue
                app_metrics.upload_cleanup_total.labels(reason="retention").inc()
                continue
            files.append((stat.st_mtime, path, stat.st_size))
    total = sum(size for _, _, size in files)
    for _, path, size in sorted(files):
        if total <= settings.upload_quota_bytes:
            break
        if path == protected:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("upload_cleanup_failed path=%s", path)
            continue
        total -= size
        app_metrics.upload_cleanup_total.labels(reason="quota").inc()
        logger.info("upload_cleanup_quota path=%s", path)


def _corpus_revision(engine: RAGEngine) -> str:
    """Read the vector collection revision without coupling the API to a plugin."""
    if not isinstance(engine.retriever, Retriever):
        return "0"
    metadata = engine.retriever.vector_store.collection.metadata or {}
    return str(metadata.get("localrag:corpus_revision", "0"))


def _provider_name(engine: RAGEngine) -> str:
    if not hasattr(engine, "provider"):
        return "ollama"
    return engine.provider.provider_name


def _default_model(engine: RAGEngine) -> str:
    if not hasattr(engine, "provider"):
        return engine.settings.ollama_llm_model
    return engine.provider.default_model


def _effective_identity(engine: RAGEngine, requested_model: str | None = None) -> tuple[str, str]:
    if not hasattr(engine, "provider"):
        return "ollama", requested_model or engine.settings.ollama_llm_model
    provider = engine.provider.effective_provider
    model = engine.provider.effective_model if requested_model is None else requested_model
    return provider, model


def query_json(  # noqa: C901, PLR0915
    request: QueryRequest, engine: RAGEngine, query_cache: QueryCache | None = None
) -> QueryResponse:
    """Blocking JSON query — retrieves context then generates a full answer."""
    t0 = time.perf_counter()
    cache_key: str | None = None
    if query_cache is not None:
        cache_key = make_cache_key(
            request.question,
            request.model or _default_model(engine),
            request.n_results,
            engine.settings.retrieval_mode,
            metadata_filter=request.metadata_filter,
            collection=engine.settings.chroma_collection_name,
            provider=_provider_name(engine),
            corpus_revision=_corpus_revision(engine),
        )
        cached = query_cache.get(cache_key)
        if cached is not None:
            logger.info("query_cache_hit")
            app_metrics.cache_operations_total.labels(operation="hit").inc()
            return QueryResponse(
                answer=str(cached["answer"]),
                sources=[SourceRef(**source) for source in cached["sources"]],
                latency_ms=float(cached["latency_ms"]),
                model=str(cached["model"]),
                provider=str(cached["provider"]),
                low_confidence=bool(cached["low_confidence"]),
                trace=cast("dict[str, object] | None", cached.get("trace")),
            )
        app_metrics.cache_operations_total.labels(operation="miss").inc()

    try:
        if engine.settings.adaptive_enabled:
            try:
                with span(SpanName.RETRIEVAL_ADAPTIVE, {"stage": "adaptive"}):
                    result = engine.answer(
                        request.question, request.model, request.n_results, request.metadata_filter
                    )
            except Exception:
                app_metrics.query_failures_total.inc()
                app_metrics.query_requests_total.labels(transport="json", outcome="error").inc()
                raise
            raw_sources = cast("list[dict[str, Any]]", result.get("sources") or [])
            response = QueryResponse(
                answer=str(result["answer"]),
                sources=[SourceRef(**dict(source)) for source in raw_sources],
                latency_ms=(time.perf_counter() - t0) * 1000,
                model=_effective_identity(engine, request.model)[1],
                provider=_effective_identity(engine, request.model)[0],
                low_confidence=not bool(raw_sources),
                trace=cast("dict[str, object] | None", result.get("trace"))
                if isinstance(result.get("trace"), dict)
                else None,
            )
            app_metrics.query_duration_seconds.observe(response.latency_ms / 1000)
            retrieved_chunks = result.get("retrieved_chunks", 0)
            app_metrics.chunks_retrieved_total.inc(
                int(retrieved_chunks) if isinstance(retrieved_chunks, (int, float)) else 0
            )
            app_metrics.tokens_used_total.labels(provider=response.provider).inc(
                len(response.answer)
            )
            write_audit_record(
                engine.settings.audit_log_path,
                correlation_id=request_id_ctx.get(""),
                question=request.question,
                sources=[asdict(s) for s in response.sources],
                answer=response.answer,
                model=response.model,
                provider=response.provider,
                latency_ms=response.latency_ms,
                max_bytes=engine.settings.audit_log_max_bytes,
                retention_seconds=engine.settings.audit_log_retention_seconds,
                metadata_only=engine.settings.audit_log_metadata_only,
                redact_content=engine.settings.audit_log_redact_content,
            )
            if query_cache is not None and cache_key is not None:
                query_cache.set(cache_key, asdict(response))
            app_metrics.query_requests_total.labels(transport="json", outcome="success").inc()
            return response
        with span(SpanName.RETRIEVAL, {"stage": "retrieve"}):
            contexts = engine.retriever.retrieve(
                question=request.question,
                n_results=request.n_results,
                metadata_filter=request.metadata_filter,
            )
    except RetrievalError as exc:
        app_metrics.query_failures_total.inc()
        app_metrics.query_requests_total.labels(transport="json", outcome="error").inc()
        kind = _query_error_kind(exc)
        raise QueryError(kind, exc.detail) from exc

    answer_chunks: list[str] = []
    low_confidence = False
    trace: dict[str, object] | None = None
    try:
        with span(SpanName.GENERATION, {"model": request.model or _default_model(engine)}):
            for event in engine.stream_chat_from_contexts(
                contexts=contexts,
                question=request.question,
                model=request.model,
            ):
                if event["type"] == "token":
                    answer_chunks.append(str(event["token"]))
                if event["type"] == "final":
                    low_confidence = bool(event.get("low_confidence", False))
                    trace = cast("dict[str, object] | None", event.get("trace"))
    except Exception:
        app_metrics.query_failures_total.inc()
        app_metrics.query_requests_total.labels(transport="json", outcome="error").inc()
        raise QueryError(ApplicationErrorKind.BAD_GATEWAY, "LLM provider request failed.") from None

    latency_ms = (time.perf_counter() - t0) * 1000
    used_provider, used_model = _effective_identity(engine, request.model)
    sources = [SourceRef(**s) for s in engine.extract_sources(contexts)]

    app_metrics.query_duration_seconds.observe(latency_ms / 1000)
    app_metrics.chunks_retrieved_total.inc(len(contexts))
    app_metrics.tokens_used_total.labels(provider=used_provider).inc(len(answer_chunks))

    logger.info(
        "query_json_done model=%s latency_ms=%.1f sources=%s",
        used_model,
        latency_ms,
        len(sources),
    )
    response = QueryResponse(
        answer="".join(answer_chunks).strip(),
        sources=sources,
        latency_ms=latency_ms,
        model=used_model,
        provider=used_provider,
        low_confidence=low_confidence,
        trace=trace,
    )
    write_audit_record(
        engine.settings.audit_log_path,
        correlation_id=request_id_ctx.get(""),
        question=request.question,
        sources=[asdict(s) for s in sources],
        answer=response.answer,
        model=used_model,
        provider=used_provider,
        latency_ms=latency_ms,
        max_bytes=engine.settings.audit_log_max_bytes,
        retention_seconds=engine.settings.audit_log_retention_seconds,
        metadata_only=engine.settings.audit_log_metadata_only,
        redact_content=engine.settings.audit_log_redact_content,
    )
    if query_cache is not None and cache_key is not None:
        query_cache.set(cache_key, asdict(response))
    app_metrics.query_requests_total.labels(transport="json", outcome="success").inc()
    return response


def _query_error_kind(exc: RetrievalError) -> ApplicationErrorKind:
    if exc.kind == RetrievalFailureKind.SERVICE_UNAVAILABLE:
        return ApplicationErrorKind.SERVICE_UNAVAILABLE
    return ApplicationErrorKind.BAD_GATEWAY


def get_query_contexts(request: QueryRequest, engine: RAGEngine) -> list[dict[str, Any]]:
    """Retrieve chunks synchronously so embedding / vector errors map to HTTP before SSE starts."""
    try:
        with span(SpanName.RETRIEVAL, {"stage": "retrieve"}):
            return engine.retriever.retrieve(
                question=request.question,
                n_results=request.n_results,
                metadata_filter=request.metadata_filter,
            )
    except RetrievalError as exc:
        kind = _query_error_kind(exc)
        raise QueryError(kind, exc.detail) from exc


def iter_query_sse_events(
    request: QueryRequest,
    engine: RAGEngine,
    contexts: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    t0 = time.perf_counter()
    logger.info(
        "query_start model=%s n_results=%s question_chars=%s",
        request.model,
        request.n_results,
        len(request.question),
    )
    answer_chunks: list[str] = []
    try:
        stream = engine.stream_chat_from_contexts(
            contexts=contexts,
            question=request.question,
            model=request.model,
        )
        for event in stream:
            if event["type"] == "token":
                token = str(event["token"])
                answer_chunks.append(token)
                yield {"event": "token", "data": token}
            if event["type"] == "final":
                latency_ms = (time.perf_counter() - t0) * 1000
                provider, model = _effective_identity(engine, request.model)
                app_metrics.query_duration_seconds.observe(latency_ms / 1000)
                app_metrics.chunks_retrieved_total.inc(len(contexts))
                app_metrics.tokens_used_total.labels(provider=provider).inc(len(answer_chunks))
                payload = {
                    "sources": event["sources"],
                    "low_confidence": event.get("low_confidence", False),
                    "trace": event.get("trace"),
                    "provider": provider,
                    "model": model,
                }
                write_audit_record(
                    engine.settings.audit_log_path,
                    correlation_id=request_id_ctx.get(""),
                    question=request.question,
                    sources=event["sources"],
                    answer="".join(answer_chunks).strip(),
                    model=model,
                    provider=provider,
                    latency_ms=latency_ms,
                    max_bytes=engine.settings.audit_log_max_bytes,
                    retention_seconds=engine.settings.audit_log_retention_seconds,
                    metadata_only=engine.settings.audit_log_metadata_only,
                    redact_content=engine.settings.audit_log_redact_content,
                )
                yield {"event": "final", "data": json.dumps(payload)}
                app_metrics.query_requests_total.labels(transport="sse", outcome="success").inc()
    except Exception:
        app_metrics.query_failures_total.inc()
        app_metrics.query_requests_total.labels(transport="sse", outcome="error").inc()
        yield {
            "event": "error",
            "data": json.dumps({"detail": "LLM provider request failed."}),
        }

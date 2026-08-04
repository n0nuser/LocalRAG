"""Privacy-first, optional OpenTelemetry tracing.

The module deliberately imports OpenTelemetry only after tracing is enabled. The
base install therefore keeps the existing offline/no-op behavior and does not
need telemetry packages. Spans contain bounded operational metadata only; query,
document, prompt, answer, token, credential, and API-key values are rejected by
default.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from typing import Any

from localrag.settings import Settings

logger = logging.getLogger(__name__)


class SpanName(StrEnum):
    API_QUERY = "localrag.api.query"
    CLI_QUERY = "localrag.cli.query"
    INGESTION = "localrag.ingestion"
    INGEST_PARSE = "localrag.ingestion.parse"
    INGEST_CHUNK = "localrag.ingestion.chunk"
    INGEST_EMBED = "localrag.ingestion.embed"
    RETRIEVAL = "localrag.retrieval"
    RETRIEVAL_VECTOR = "localrag.retrieval.vector"
    RETRIEVAL_BM25 = "localrag.retrieval.bm25"
    RETRIEVAL_RRF = "localrag.retrieval.rrf"
    RETRIEVAL_FRESHNESS = "localrag.retrieval.freshness"
    RETRIEVAL_RERANK = "localrag.retrieval.rerank"
    RETRIEVAL_COMPRESSION = "localrag.retrieval.compression"
    RETRIEVAL_ADAPTIVE = "localrag.retrieval.adaptive"
    GENERATION = "localrag.generation"
    BENCHMARK = "localrag.benchmark"
    EVALUATION = "localrag.evaluation"


_SENSITIVE_NAMES = {
    "answer",
    "api_key",
    "credential",
    "document",
    "prompt",
    "query",
    "question",
    "text",
    "token",
}
_ALLOWED_NAMES = {
    "batch_size",
    "collection",
    "count",
    "duration_ms",
    "error_type",
    "file_type",
    "model",
    "provider",
    "request_id",
    "run_id",
    "stage",
    "status",
    "tenant_id",
}


def _safe_attributes(settings: Settings, attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    output: dict[str, Any] = {}
    limit = settings.otel_max_attribute_length
    for name, value in attributes.items():
        normalized = name.lower().replace("-", "_")
        if (normalized in _SENSITIVE_NAMES or normalized.endswith(("_key", "_secret"))) and not (
            settings.otel_capture_content and normalized in {"prompt", "document", "query"}
        ):
            continue
        if normalized not in _ALLOWED_NAMES and not (
            settings.otel_capture_content and normalized in {"prompt", "document", "query"}
        ):
            continue
        bounded_value: Any
        if isinstance(value, str):
            bounded_value = value[:limit]
        elif isinstance(value, (bool, int, float)):
            bounded_value = value
        else:
            continue
        output[f"localrag.{normalized}"] = bounded_value
    return output


class Tracing:
    """Process-local tracer lifecycle and safe span factory."""

    _tracer: Any = None
    _provider: Any = None
    _settings = Settings()

    @classmethod
    def configure(cls, settings: Settings, provider: Any = None) -> None:
        cls.shutdown()
        cls._settings = settings
        if not settings.otel_enabled:
            return
        try:
            if provider is None:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
                from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
                from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
                from opentelemetry.sdk.trace.sampling import TraceIdRatioBased  # noqa: PLC0415

                provider = TracerProvider(
                    resource=Resource.create({"service.name": settings.otel_service_name}),
                    sampler=TraceIdRatioBased(settings.otel_sample_rate),
                )
                exporter = OTLPSpanExporter(
                    endpoint=settings.otel_exporter_endpoint.rstrip("/") + "/v1/traces",
                    timeout=int(settings.otel_exporter_timeout_seconds),
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
            cls._provider = provider
            cls._tracer = (
                None if settings.otel_sample_rate == 0 else provider.get_tracer("localrag")
            )
        except ImportError:
            logger.warning("otel_enabled_but_optional_dependency_missing")
        except Exception:
            logger.exception("otel_initialization_failed")
            cls._tracer = None
            cls._provider = None

    @classmethod
    def shutdown(cls) -> None:
        provider, cls._provider = cls._provider, None
        cls._tracer = None
        if provider is None:
            return
        timeout = int(cls._settings.otel_exporter_timeout_seconds * 1000)
        try:
            provider.force_flush(timeout_millis=timeout)
        except Exception:
            logger.warning("otel_flush_failed", exc_info=True)
        try:
            provider.shutdown()
        except Exception:
            logger.warning("otel_shutdown_failed", exc_info=True)

    @classmethod
    def current_span_id(cls) -> str | None:
        if cls._tracer is None:
            return None
        try:
            from opentelemetry import trace  # noqa: PLC0415

            context = trace.get_current_span().get_span_context()
            return format(context.span_id, "016x") if context.is_valid else None
        except Exception:
            return None


@contextmanager
def span(name: SpanName | str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Create a safe span, never allowing telemetry failures to affect work."""
    tracer = Tracing._tracer  # noqa: SLF001
    if tracer is None:
        with nullcontext() as current:
            yield current
        return
    try:
        context_manager = tracer.start_as_current_span(
            str(name),
            attributes=_safe_attributes(Tracing._settings, attributes),  # noqa: SLF001
        )
    except Exception:
        logger.warning("otel_span_start_failed name=%s", name, exc_info=True)
        with nullcontext() as current:
            yield current
        return
    with context_manager as current:
        yield current


def configure_tracing(settings: Settings, provider: Any = None) -> None:
    Tracing.configure(settings, provider)


def shutdown_tracing() -> None:
    Tracing.shutdown()

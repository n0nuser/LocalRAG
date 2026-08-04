from __future__ import annotations

from typing import Any

import pytest

from localrag.observability.tracing import (
    SpanName,
    Tracing,
    configure_tracing,
    shutdown_tracing,
    span,
)
from localrag.settings import Settings


def _memory_provider() -> tuple[Any, Any]:
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_disabled_tracing_is_a_noop_without_content() -> None:
    configure_tracing(Settings())

    with span(SpanName.API_QUERY, {"question": "secret question", "model": "local"}):
        pass

    assert Tracing.current_span_id() is None
    shutdown_tracing()


def test_spans_have_safe_attributes_and_parent_relationship() -> None:
    provider, exporter = _memory_provider()
    configure_tracing(Settings(otel_enabled=True), provider=provider)

    with (
        span(SpanName.API_QUERY, {"request_id": "req-1", "question": "secret"}),
        span(SpanName.RETRIEVAL_VECTOR, {"model": "embed", "count": 3}),
    ):
        pass

    spans = exporter.get_finished_spans()
    assert [item.name for item in spans] == [SpanName.RETRIEVAL_VECTOR, SpanName.API_QUERY]
    child, parent = spans
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
    assert parent.attributes["localrag.request_id"] == "req-1"
    assert "localrag.question" not in parent.attributes
    assert child.attributes["localrag.count"] == 3
    assert "localrag.model" in child.attributes
    shutdown_tracing()


def test_content_capture_is_explicit_and_bounded() -> None:
    provider, exporter = _memory_provider()
    configure_tracing(
        Settings(otel_enabled=True, otel_capture_content=True, otel_max_attribute_length=8),
        provider=provider,
    )

    with span(SpanName.GENERATION, {"prompt": "0123456789-secret"}):
        pass

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["localrag.prompt"] == "01234567"
    shutdown_tracing()


def test_failed_work_marks_span_error_without_changing_exception() -> None:
    provider, exporter = _memory_provider()
    configure_tracing(Settings(otel_enabled=True), provider=provider)

    with pytest.raises(RuntimeError, match="boom"), span(SpanName.GENERATION, {"stage": "test"}):
        raise RuntimeError("boom")

    failed = exporter.get_finished_spans()[0]
    assert failed.status.status_code.name == "ERROR"
    shutdown_tracing()


def test_sampling_zero_does_not_export() -> None:
    provider, exporter = _memory_provider()
    configure_tracing(Settings(otel_enabled=True, otel_sample_rate=0), provider=provider)

    with span(SpanName.GENERATION, {"model": "local"}):
        pass

    assert not exporter.get_finished_spans()
    shutdown_tracing()


def test_exporter_failure_never_escapes_and_shutdown_flushes() -> None:
    class BrokenProvider:
        def get_tracer(self, _name: str) -> Any:
            provider, _ = _memory_provider()
            return provider.get_tracer("test")

        def force_flush(self, timeout_millis: int) -> bool:
            _ = timeout_millis
            raise RuntimeError("flush failed")

        def shutdown(self) -> None:
            raise RuntimeError("shutdown failed")

    configure_tracing(Settings(otel_enabled=True), provider=BrokenProvider())
    with (
        pytest.raises(RuntimeError, match="application"),
        span(SpanName.GENERATION, {"model": "local"}),
    ):
        raise RuntimeError("application")
    shutdown_tracing()

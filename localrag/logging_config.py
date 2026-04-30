"""Configure structured logging for the ``localrag`` logger tree (CLI and API).

Production/CI → JSON via structlog; local dev → human-readable console renderer.
Set ``LOG_FORMAT=json`` to force JSON even in a terminal, or ``LOG_FORMAT=console``
to force human-readable output.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

import structlog

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def _add_request_id(
    _logger: object,
    _method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    event_dict["correlation_id"] = request_id_ctx.get()
    return event_dict


def _parse_level(name: str) -> int:
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def _use_json_renderer() -> bool:
    fmt = os.environ.get("LOG_FORMAT", "").lower()
    if fmt == "json":
        return True
    if fmt == "console":
        return False
    return not sys.stderr.isatty()


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging.

    Safe to call more than once; only attaches the stdlib handler once.
    Always updates log level on subsequent calls.
    """
    numeric = _parse_level(level)
    use_json = _use_json_renderer()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_request_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if use_json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    localrag_log = logging.getLogger("localrag")
    localrag_log.setLevel(numeric)

    if not localrag_log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(numeric)
        handler.setFormatter(formatter)
        localrag_log.addHandler(handler)
        localrag_log.propagate = False
    else:
        for h in localrag_log.handlers:
            h.setLevel(numeric)
            h.setFormatter(formatter)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

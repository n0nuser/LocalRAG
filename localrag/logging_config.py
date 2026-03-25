"""Configure the ``localrag`` logger tree for CLI and API."""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Attach ``request_id`` for format strings (API middleware sets the context var)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def _parse_level(name: str) -> int:
    level = getattr(logging, name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def configure_logging(level: str = "INFO") -> None:
    """Attach a stderr handler to ``localrag`` and tune third-party noise.

    Safe to call more than once: adds the handler only once, always updates level.
    """
    numeric = _parse_level(level)
    hermit_log = logging.getLogger("localrag")
    hermit_log.setLevel(numeric)

    if not hermit_log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(numeric)
        handler.addFilter(RequestIdFilter())
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        hermit_log.addHandler(handler)
        hermit_log.propagate = False
    else:
        for h in hermit_log.handlers:
            h.setLevel(numeric)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

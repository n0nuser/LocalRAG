from __future__ import annotations

import logging

from localrag.logging_config import _parse_level, configure_logging, request_id_ctx


def test_request_id_ctx_default() -> None:
    assert request_id_ctx.get() == "-"


def test_request_id_ctx_set_and_reset() -> None:
    token = request_id_ctx.set("req-abc")
    assert request_id_ctx.get() == "req-abc"
    request_id_ctx.reset(token)
    assert request_id_ctx.get() == "-"


def test_parse_level_falls_back_to_info_on_unknown() -> None:
    assert _parse_level("NOT_A_LEVEL") == logging.INFO
    assert _parse_level("error") == logging.ERROR


def test_configure_logging_is_idempotent() -> None:
    configure_logging("INFO")
    hermit_log = logging.getLogger("localrag")
    initial = len(hermit_log.handlers)
    configure_logging("ERROR")
    assert len(hermit_log.handlers) == initial

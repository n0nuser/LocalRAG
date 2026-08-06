from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette

from localrag.application.container import (
    get_collection_repository,
    get_engine,
    get_ingestion_service,
    get_retriever,
)
from localrag.application.runtime import get_embedder
from localrag.mcp.server import build_mcp_server
from localrag.settings import get_settings, load_settings, set_current_settings


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncIterator[None]:
    set_current_settings(load_settings(os.environ.get("LOCALRAG_CONFIG")))
    try:
        yield
    finally:
        if get_embedder.cache_info().currsize:
            get_embedder().close()
        if get_retriever.cache_info().currsize:
            get_retriever().close()


def _build_app() -> Any:
    """Build the FastMCP ASGI app.

    ``get_engine`` / ``get_ingestion_service`` / ``get_collection_repository``
    are passed as factories rather than called here: they are ``lru_cache``d,
    so the first real tool call builds the embedder/vector store/retriever
    once and every later call (and every ``tools/list``/``initialize``) is
    free. This keeps startup cheap, matching the previous hand-rolled adapter's
    behavior of only loading dependencies for ``tools/call``.

    ``FastMCP.http_app()`` has no ``lifespan=`` parameter of its own, so the
    settings-loading/dependency-teardown lifespan is passed into
    ``build_mcp_server`` and forwarded to the ``FastMCP`` constructor instead,
    which is what ``http_app()``'s returned ASGI app actually runs.
    """
    mcp = build_mcp_server(
        settings=get_settings(),
        engine_factory=get_engine,
        ingestion_service_factory=get_ingestion_service,
        collection_repo_factory=get_collection_repository,
        lifespan=lifespan,
    )
    return mcp.http_app(path="/mcp", stateless_http=True)


app = _build_app()

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from localrag.application.container import (
    get_collection_repository,
    get_engine,
    get_ingestion_service,
    get_retriever,
)
from localrag.application.runtime import get_embedder
from localrag.mcp.server import McpServer
from localrag.settings import get_settings, load_settings, set_current_settings


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    set_current_settings(load_settings(os.environ.get("LOCALRAG_CONFIG")))
    try:
        yield
    finally:
        if get_embedder.cache_info().currsize:
            get_embedder().close()
        if get_retriever.cache_info().currsize:
            get_retriever().close()


app = FastAPI(title="LocalRAG MCP", version="0.1.0", lifespan=lifespan)


def _server(load_dependencies: bool) -> McpServer:
    if not load_dependencies:
        return McpServer(settings=get_settings())
    return McpServer(
        settings=get_settings(),
        engine=get_engine(),
        ingestion_service=get_ingestion_service(),
        collection_repo=get_collection_repository(),
    )


@app.post("/mcp")
async def mcp(request: Request) -> JSONResponse:
    message = await request.json()
    api_key = request.headers.get("X-API-Key")
    response = _server(message.get("method") == "tools/call").handle_message(message, api_key)
    if response is None:
        return JSONResponse(content={})
    return JSONResponse(content=response)


def handle_stdio_message(
    message: dict[str, Any], api_key: str | None = None
) -> dict[str, Any] | None:
    return _server(message.get("method") == "tools/call").handle_message(message, api_key)

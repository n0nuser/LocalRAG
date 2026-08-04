from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from localrag.api.dependencies import get_embedder, get_retriever
from localrag.api.exceptions import HttpMappedError
from localrag.api.middleware import RequestContextMiddleware
from localrag.api.routers.agent import router as agent_router
from localrag.api.routers.collections import router as collections_router
from localrag.api.routers.health import router as health_router
from localrag.api.routers.ingest import router as ingest_router
from localrag.api.routers.metrics import router as metrics_router
from localrag.api.routers.query import router as query_router
from localrag.logging_config import configure_logging
from localrag.observability.tracing import configure_tracing, shutdown_tracing
from localrag.settings import get_settings, load_settings, set_current_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    set_current_settings(load_settings(os.environ.get("LOCALRAG_CONFIG")))
    configure_logging(get_settings().log_level)
    configure_tracing(get_settings())
    logger.info("api_startup")
    try:
        yield
    finally:
        if get_embedder.cache_info().currsize:
            get_embedder().close()
        if get_retriever.cache_info().currsize:
            get_retriever().close()  # type: ignore[attr-defined]
        shutdown_tracing()
        logger.info("api_shutdown")


app = FastAPI(title="LocalRAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(agent_router)
app.include_router(collections_router)


@app.exception_handler(HttpMappedError)
async def http_mapped_error_handler(request: Request, exc: HttpMappedError) -> JSONResponse:
    logger.warning(
        "http_mapped_error %s %s status=%s detail=%s",
        request.method,
        request.url.path,
        exc.status_code,
        exc.detail,
    )
    return JSONResponse(
        status_code=int(exc.status_code),
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "unhandled_exception %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning(
        "validation_error %s %s errors=%s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": exc.errors()},
    )

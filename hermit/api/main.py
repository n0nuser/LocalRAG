from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from hermit.api.exceptions import IngestApiError
from hermit.api.middleware import RequestContextMiddleware
from hermit.api.routers.collections import router as collections_router
from hermit.api.routers.health import router as health_router
from hermit.api.routers.ingest import router as ingest_router
from hermit.api.routers.query import router as query_router
from hermit.logging_config import configure_logging
from hermit.settings import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    logger.info("api_startup")
    yield
    logger.info("api_shutdown")


app = FastAPI(title="Hermit API", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestContextMiddleware)

app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(collections_router)


@app.exception_handler(IngestApiError)
async def ingest_api_error_handler(request: Request, exc: IngestApiError) -> JSONResponse:
    logger.warning(
        "ingest_api_error %s %s status=%s detail=%s",
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

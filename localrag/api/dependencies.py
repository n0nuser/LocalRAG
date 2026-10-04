from __future__ import annotations

from http import HTTPStatus

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from localrag.application.container import Container
from localrag.application.jobs import JobRegistry
from localrag.application.repository import ChromaCollectionRepository
from localrag.rag.engine import RAGEngine
from localrag.rag.query_cache import QueryCache

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_container(request: Request) -> Container:
    """The process container the lifespan built; tests override this one dependency."""
    return request.app.state.container


def get_engine(container: Container = Depends(get_container)) -> RAGEngine:
    return container.engine


def get_query_cache(container: Container = Depends(get_container)) -> QueryCache:
    return container.query_cache


def get_job_registry(container: Container = Depends(get_container)) -> JobRegistry:
    return container.job_registry


def get_collection_repository(
    container: Container = Depends(get_container),
) -> ChromaCollectionRepository:
    return container.collection_repository


def require_api_key(
    key: str | None = Security(_api_key_header),
    container: Container = Depends(get_container),
) -> None:
    """Enforce X-API-Key when API_KEY is configured. No-op when API_KEY is empty."""
    configured = container.settings.api_key
    if not configured:
        return
    if not key or key != configured:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from mcp import types as mt
from pydantic import Field

from localrag.application import service as application_service
from localrag.application.dto import IngestDirectoryRequest, IngestFileRequest, QueryRequest
from localrag.application.errors import ApplicationError
from localrag.application.repository import ChromaCollectionRepository
from localrag.ingestion.service import IngestionService
from localrag.rag.engine import RAGEngine
from localrag.settings import Settings

SERVER_NAME = "localrag"


def _dependency_accessor[T](
    factory: Callable[[], T] | None, missing_message: str
) -> Callable[[], T]:
    """Build a lazy accessor that raises with a clear message when unconfigured.

    Keeps each dependency's "not configured" ``RuntimeError`` guard identical
    to the previous hand-rolled adapter, without repeating the same three
    lines once per dependency in ``build_mcp_server``.
    """

    def accessor() -> T:
        if factory is None:
            raise RuntimeError(missing_message)
        return factory()

    return accessor


class ApiKeyMiddleware(Middleware):
    """Rejects tool calls when the configured API key does not match the caller's.

    HTTP transport carries the key in the ``X-API-Key`` header; stdio has no
    headers, so it falls back to the ``MCP_API_KEY`` environment variable to
    match the API key contract the hand-rolled adapter used before FastMCP.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, Any],
    ) -> Any:
        if self.settings.api_key and self._caller_api_key() != self.settings.api_key:
            raise ToolError("Invalid or missing API key.")
        return await call_next(context)

    @staticmethod
    def _caller_api_key() -> str | None:
        headers = get_http_headers()
        if headers:
            return headers.get("x-api-key")
        return os.environ.get("MCP_API_KEY")


def build_mcp_server(
    settings: Settings,
    engine_factory: Callable[[], RAGEngine] | None = None,
    ingestion_service_factory: Callable[[], IngestionService] | None = None,
    collection_repo_factory: Callable[[], ChromaCollectionRepository] | None = None,
    lifespan: Callable[[Any], AbstractAsyncContextManager[None]] | None = None,
) -> FastMCP:
    """Build the FastMCP server with the four LocalRAG tools registered.

    Each dependency is a factory rather than an instance so production callers
    (see ``localrag/mcp/app.py``) can defer building the embedder, vector
    store, and retriever until a tool actually runs. ``tools/list`` and
    ``initialize`` never call these factories, so startup stays cheap. Tests
    inject factories that return stubs.

    ``lifespan`` is forwarded to the underlying ``FastMCP`` server: it is the
    only way an ASGI lifespan set here actually runs under ``http_app()``,
    which does not accept a ``lifespan=`` argument of its own.
    """
    mcp = FastMCP(name=SERVER_NAME, middleware=[ApiKeyMiddleware(settings)], lifespan=lifespan)

    _engine = _dependency_accessor(engine_factory, "MCP engine is not configured.")
    _ingestion_service = _dependency_accessor(
        ingestion_service_factory, "MCP ingestion service is not configured."
    )
    _collection_repo = _dependency_accessor(
        collection_repo_factory, "MCP collection repository is not configured."
    )

    @mcp.tool
    def search_documents(
        question: Annotated[str, Field(description="Question to search for.")],
        n_results: Annotated[
            int | None, Field(default=None, ge=1, description="Number of chunks to return.")
        ] = None,
        metadata_filter: Annotated[
            dict[str, str] | None,
            Field(default=None, description="Optional metadata equality filters."),
        ] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant document chunks without generating an answer."""
        request = QueryRequest(
            question=question, n_results=n_results, metadata_filter=metadata_filter
        )
        return _run_tool(lambda: application_service.get_query_contexts(request, _engine()))

    @mcp.tool
    def answer_question(
        question: Annotated[str, Field(description="Question to answer.")],
        model: Annotated[
            str | None, Field(default=None, description="Override the default LLM model.")
        ] = None,
        n_results: Annotated[
            int | None, Field(default=None, ge=1, description="Number of chunks to retrieve.")
        ] = None,
        metadata_filter: Annotated[
            dict[str, str] | None,
            Field(default=None, description="Optional metadata equality filters."),
        ] = None,
    ) -> dict[str, Any]:
        """Answer a question from the ingested document collection with citations."""
        request = QueryRequest(
            question=question,
            model=model,
            n_results=n_results,
            metadata_filter=metadata_filter,
        )
        return _run_tool(lambda: asdict(application_service.query_json(request, _engine())))

    @mcp.tool
    def ingest_path(
        path: Annotated[str, Field(description="Allowed file or directory path to ingest.")],
        recursive: Annotated[
            bool | None, Field(default=None, description="Recurse into subdirectories.")
        ] = None,
        embed_model: Annotated[
            str | None, Field(default=None, description="Override the default embedding model.")
        ] = None,
    ) -> dict[str, Any]:
        """Ingest an allowed file or directory into the document collection."""

        def call() -> dict[str, Any]:
            if Path(path).is_dir():
                return asdict(
                    application_service.ingest_directory(
                        IngestDirectoryRequest(
                            path=path, recursive=recursive, embed_model=embed_model
                        ),
                        settings,
                        _ingestion_service(),
                    )
                )
            return asdict(
                application_service.ingest_file(
                    IngestFileRequest(path=path, embed_model=embed_model),
                    settings,
                    _ingestion_service(),
                )
            )

        return _run_tool(call)

    @mcp.tool
    def list_collections() -> dict[str, Any]:
        """List available document collections."""
        return _run_tool(
            lambda: asdict(application_service.list_collections_response(_collection_repo()))
        )

    return mcp


def _run_tool(call: Callable[[], Any]) -> Any:
    """Map application/domain errors to ``ToolError`` the same way the old adapter did."""
    try:
        return call()
    except ApplicationError as exc:
        raise ToolError(exc.detail) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError(str(exc)) from exc

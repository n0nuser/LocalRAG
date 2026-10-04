from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_context, get_http_headers
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from pydantic import Field

from localrag.application import service as application_service
from localrag.application.container import Container
from localrag.application.dto import IngestDirectoryRequest, IngestFileRequest, QueryRequest
from localrag.application.errors import ApplicationError
from localrag.settings import Settings

SERVER_NAME = "localrag"
_CONTAINER = "container"


class ApiKeyMiddleware(Middleware):
    """Rejects requests when the configured API key does not match the caller's.

    HTTP transport carries the key in the ``X-API-Key`` header; stdio has no
    headers, so it falls back to the ``MCP_API_KEY`` environment variable to
    match the API key contract the hand-rolled adapter used before FastMCP.

    The check hooks ``on_request`` rather than ``on_call_tool`` so that
    ``tools/list`` and ``initialize`` stay gated too. The previous hand-rolled
    adapter rejected an unauthenticated caller on *every* method, and the tool
    list is itself information about the deployment worth withholding.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def on_request(
        self,
        context: MiddlewareContext[Any],
        call_next: CallNext[Any, Any],
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


def build_mcp_server(settings: Settings, build_container: Callable[[], Container]) -> FastMCP:
    """Build the FastMCP server with the four LocalRAG tools registered.

    The server lifespan calls ``build_container`` once per process and closes the
    container on exit; tools reach it through the lifespan context, which is how
    FastMCP shares process-lifetime resources. Under stdio the lifespan runs inside
    ``mcp.run``, under HTTP inside the ASGI app's lifespan (``http_app()`` takes no
    ``lifespan=`` of its own). The container builds nothing until a tool needs it,
    so ``initialize`` and ``tools/list`` stay cheap.
    """

    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Container]]:
        container = build_container()
        try:
            yield {_CONTAINER: container}
        finally:
            container.close()

    mcp = FastMCP(name=SERVER_NAME, middleware=[ApiKeyMiddleware(settings)], lifespan=lifespan)

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
        return _run_tool(
            lambda: application_service.get_query_contexts(request, _container().engine)
        )

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
        return _run_tool(
            lambda: asdict(application_service.query_json(request, _container().engine))
        )

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
            container = _container()
            if Path(path).is_dir():
                result = asdict(
                    application_service.ingest_directory(
                        IngestDirectoryRequest(
                            path=path, recursive=recursive, embed_model=embed_model
                        ),
                        container.settings,
                        container.ingestion_service,
                    )
                )
            else:
                result = asdict(
                    application_service.ingest_file(
                        IngestFileRequest(path=path, embed_model=embed_model),
                        container.settings,
                        container.ingestion_service,
                    )
                )
            container.invalidate()
            return result

        return _run_tool(call)

    @mcp.tool
    def list_collections() -> dict[str, Any]:
        """List available document collections."""
        return _run_tool(
            lambda: asdict(
                application_service.list_collections_response(_container().collection_repository)
            )
        )

    return mcp


def _container() -> Container:
    """The container this server's lifespan built, for the tool call in progress."""
    return get_context().lifespan_context[_CONTAINER]


def _run_tool(call: Callable[[], Any]) -> Any:
    """Map application/domain errors to ``ToolError`` the same way the old adapter did."""
    try:
        return call()
    except ApplicationError as exc:
        raise ToolError(exc.detail) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ToolError(str(exc)) from exc

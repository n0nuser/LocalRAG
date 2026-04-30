from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse

from localrag.api import service as api_service
from localrag.api.dependencies import get_engine, require_api_key
from localrag.api.schemas import QueryRequest, QueryResponse
from localrag.rag.engine import RAGEngine

router = APIRouter(prefix="", tags=["query"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=QueryResponse, summary="Query (JSON)")
def query(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> QueryResponse:
    """Retrieve context and return a complete JSON response with answer, sources, and latency."""
    return api_service.query_json(request, engine)


@router.post("/query/stream", summary="Query (SSE stream)")
def query_stream(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> EventSourceResponse:
    """Stream answer tokens via Server-Sent Events. Final event includes source references."""
    contexts = api_service.get_query_contexts(request, engine)
    return EventSourceResponse(api_service.iter_query_sse_events(request, engine, contexts))

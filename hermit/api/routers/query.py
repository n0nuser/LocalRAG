from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse

from hermit.api import service as api_service
from hermit.api.dependencies import get_engine
from hermit.api.schemas import QueryRequest
from hermit.rag.engine import RAGEngine

router = APIRouter(prefix="", tags=["query"])


@router.post("/query")
def query(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> EventSourceResponse:
    return EventSourceResponse(api_service.iter_query_sse_events(request, engine))

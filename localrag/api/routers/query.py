from __future__ import annotations

from fastapi import APIRouter, Depends
from sse_starlette import EventSourceResponse

from localrag.api import service as api_service
from localrag.api.dependencies import get_engine
from localrag.api.schemas import QueryRequest
from localrag.rag.engine import RAGEngine

router = APIRouter(prefix="", tags=["query"])


@router.post("/query")
def query(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> EventSourceResponse:
    contexts = api_service.get_query_contexts(request, engine)
    return EventSourceResponse(api_service.iter_query_sse_events(request, engine, contexts))

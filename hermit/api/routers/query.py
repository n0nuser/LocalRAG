from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from hermit.api.dependencies import get_engine
from hermit.rag.engine import RAGEngine

router = APIRouter(prefix="", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    model: str | None = None
    n_results: int | None = None


@router.post("/query")
def query(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> EventSourceResponse:
    def stream() -> object:
        for event in engine.stream_answer(
            question=request.question,
            model=request.model,
            n_results=request.n_results,
        ):
            if event["type"] == "token":
                yield {"event": "token", "data": str(event["token"])}
            if event["type"] == "final":
                payload = {"sources": event["sources"]}
                yield {"event": "final", "data": json.dumps(payload)}

    return EventSourceResponse(stream())

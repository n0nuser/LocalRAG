from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends, HTTPException

from localrag.agent.service import run_agent
from localrag.api.dependencies import get_api_settings, get_engine, require_api_key
from localrag.api.schemas import AgentQueryRequest, AgentQueryResponse, SourceRef
from localrag.rag.engine import RAGEngine
from localrag.settings import Settings

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=AgentQueryResponse, summary="Agent query (tool-use)")
def agent_query(
    request: AgentQueryRequest,
    engine: RAGEngine = Depends(get_engine),
    settings: Settings = Depends(get_api_settings),
) -> AgentQueryResponse:
    """Run the Anthropic tool-use agent.

    The agent decides whether to search documents or answer directly.
    """
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY is not configured. Set it in .env to use the agent endpoint.",
        )
    model = request.model or settings.agent_model
    result = run_agent(
        question=request.question,
        engine=engine,
        api_key=settings.anthropic_api_key,
        model=model,
    )
    sources = [
        SourceRef(source=str(s.get("source", "")), chunk_index=int(s.get("chunk_index", -1)))  # type: ignore[call-overload]
        for s in result.sources
    ]
    return AgentQueryResponse(
        answer=result.answer,
        tool_used=result.tool_used,
        reasoning=result.reasoning,
        sources=sources,
        latency_ms=result.latency_ms,
        model=result.model,
    )

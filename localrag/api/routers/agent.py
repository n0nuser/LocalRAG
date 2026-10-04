from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, Depends

from localrag.agent.service import run_agent
from localrag.api.dependencies import get_container, require_api_key
from localrag.api.exceptions import AgentApiError
from localrag.api.schemas import AgentQueryRequest, AgentQueryResponse, SourceRef
from localrag.application.container import Container

router = APIRouter(prefix="/agent", tags=["agent"], dependencies=[Depends(require_api_key)])


@router.post("/query", response_model=AgentQueryResponse, summary="Agent query (tool-use)")
def agent_query(
    request: AgentQueryRequest,
    container: Container = Depends(get_container),
) -> AgentQueryResponse:
    """Run the Anthropic tool-use agent.

    The agent decides whether to search documents or answer directly.
    """
    settings = container.settings
    if not settings.anthropic_api_key:
        raise AgentApiError(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="ANTHROPIC_API_KEY is not configured. Set it in .env to use the agent endpoint.",
        )
    model = request.model or settings.agent_model
    result = run_agent(
        question=request.question,
        engine=container.engine,
        api_key=settings.anthropic_api_key,
        model=model,
    )
    sources = [SourceRef(**s) for s in result.sources]
    return AgentQueryResponse(
        answer=result.answer,
        tool_used=result.tool_used,
        reasoning=result.reasoning,
        sources=sources,
        latency_ms=result.latency_ms,
        model=result.model,
        trace=result.trace,
    )

"""Agent service using Anthropic tool use to decide retrieval vs. direct answer.

The agent receives a question and chooses between two tools:
- ``search_documents`` — retrieve from ChromaDB and generate a RAG-grounded answer
- ``answer_directly`` — respond without retrieval (greetings, out-of-scope, simple factual)

The ``reasoning`` field in the response explains which path was taken and why.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

from localrag.rag.engine import RAGEngine

logger = logging.getLogger(__name__)

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": (
            "Search the ingested document collection for relevant context, "
            "then use that context to generate a grounded answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to run against the document store.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "answer_directly",
        "description": (
            "Respond to the user without searching documents. "
            "Use this for greetings, out-of-scope questions, or when retrieval is unnecessary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The direct answer to give the user.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Short explanation of why no document search is needed.",
                },
            },
            "required": ["answer", "reasoning"],
        },
    },
]

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a local document search tool. "
    "For questions about the documents in the knowledge base, use search_documents. "
    "For greetings, off-topic questions, or things clearly not in the documents, "
    "use answer_directly. Always use exactly one tool."
)


@dataclass
class AgentResponse:
    """Unified response from the agent endpoint."""

    answer: str
    tool_used: str
    reasoning: str
    sources: list[dict[str, object]] = field(default_factory=list)
    latency_ms: float = 0.0
    model: str = ""


def run_agent(
    question: str,
    engine: RAGEngine,
    api_key: str,
    model: str = "claude-haiku-4-5",
) -> AgentResponse:
    """Run the Anthropic tool-use agent and return a structured response."""
    t0 = time.perf_counter()
    client = anthropic.Anthropic(api_key=api_key)

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=_TOOLS,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": question}],
    )

    tool_used = "unknown"
    reasoning = ""
    answer = ""
    sources: list[dict[str, object]] = []

    for block in resp.content:
        if block.type != "tool_use":
            continue

        tool_used = block.name
        tool_input: dict[str, Any] = block.input  # type: ignore[assignment]

        if block.name == "search_documents":
            query = str(tool_input.get("query", question))
            logger.info("agent_search_documents query=%s", query[:80])
            result: dict[str, Any] = engine.answer(question=query)
            answer = str(result.get("answer", ""))
            sources = [dict(s) for s in result.get("sources") or []]
            reasoning = f"Used search_documents with query: {query!r}"

        elif block.name == "answer_directly":
            answer = str(tool_input.get("answer", ""))
            reasoning = str(tool_input.get("reasoning", ""))
            logger.info("agent_answer_directly reasoning=%s", reasoning[:80])

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "agent_done tool=%s latency_ms=%.1f sources=%s",
        tool_used,
        latency_ms,
        len(sources),
    )
    return AgentResponse(
        answer=answer,
        tool_used=tool_used,
        reasoning=reasoning,
        sources=sources,
        latency_ms=latency_ms,
        model=model,
    )

# ADR 003: Agent Framework — Anthropic Tool Use

**Status:** Accepted  
**Date:** 2026-04-30

## Context

Phase 5 adds an agentic layer that decides, per query, whether to search documents (`search_documents`) or answer directly (`answer_directly`). The agent needs a reliable tool-calling mechanism and a model that follows instructions precisely.

Options considered:

| Option | Notes |
| --- | --- |
| **Anthropic SDK tool use** | Native tool calling via `messages.create(tools=[...])` |
| LangChain agents | Abstracts over many LLMs; adds heavyweight dependency |
| LlamaIndex agents | Similar abstraction layer; heavier than needed |
| OpenAI function calling | Well-supported but locks into OpenAI |
| Raw prompt engineering | Fragile; brittle JSON parsing |

## Decision

Implement the agent using the **Anthropic Python SDK** (`anthropic` package) with explicit `tools` parameter and `ToolUseBlock` response parsing. Exposed at `POST /agent/query`.

## Rationale

- **Reliability**: Anthropic's tool-use protocol produces clean, structured `ToolUseBlock` responses that are straightforward to parse without regex or JSON extraction heuristics.
- **Minimal dependency surface**: the `anthropic` SDK was already required for the `AnthropicProvider` LLM backend. No additional framework dependency is needed.
- **Transparency**: the `reasoning` field in `AgentResponse` records exactly which tool was chosen and why, making agent decisions auditable.
- **Two-tool simplicity**: the `search_documents` / `answer_directly` pattern is the 80/20 solution — complex multi-hop chains are out of scope for this project stage.

## Consequences

- **Positive**: predictable tool dispatch; easy to add more tools by extending `_TOOLS` in `localrag/agent/service.py`; no framework lock-in.
- **Negative**: `POST /agent/query` requires `ANTHROPIC_API_KEY` and incurs cloud API costs. The endpoint returns HTTP 503 when the key is not configured.
- **Future**: if local tool-use becomes available (e.g. via Ollama models with structured output), the agent service can be extended to use `OllamaProvider` without changing the router or schema.

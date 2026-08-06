# ADR 038: Transport-Agnostic Application Boundary and MCP Adapter

## Status

Accepted. Amended by [ADR 039](039-fastmcp-sdk-adoption.md): the MCP wire implementation described below (a hand-rolled JSON-RPC dispatcher, no SDK dependency) was replaced with the FastMCP SDK. The application-boundary decision — `localrag/application/`, transports depending on it instead of `localrag.api`, the four tool names, and the `INGEST_ROOTS`/API-key contracts — is still in force.

## Context

The REST API previously owned both HTTP adaptation and application use cases.
The CLI and retriever plugin therefore imported from `localrag.api`, and use-case errors carried HTTP status codes.
MCP needs to call the same use cases without importing FastAPI or encoding HTTP behavior.

## Decision

Transport-neutral DTOs, errors, jobs, repositories, runtime factories, and use cases live under `localrag/application/`.
The REST API maps application errors to HTTP responses and maps HTTP schemas to application DTOs.
The CLI and plugins use the application container directly.
The MCP adapter exposes only `search_documents`, `answer_question`, `ingest_path`, and `list_collections`.
MCP supports JSON-RPC over stdio and a separate HTTP `/mcp` process.
The HTTP MCP process uses the configured `API_KEY` through `X-API-Key`; stdio uses `MCP_API_KEY`.
All transports call the same ingest-root validation and persistence boundaries.

## Consequences

New transports must depend on `localrag.application`, not `localrag.api`.
The API schema layer remains responsible for OpenAPI metadata and HTTP status encoding.
The shared Docker image can run the REST and MCP entrypoints independently.
The MCP wire implementation is deliberately limited until a future issue requires broader protocol features.

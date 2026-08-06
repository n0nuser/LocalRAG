# ADR 039: Adopt the FastMCP SDK for the MCP Adapter

## Status

Accepted. Amends [ADR 038](038-application-and-mcp-boundaries.md), which described the hand-rolled JSON-RPC wire implementation this ADR replaces.

## Context

ADR 038 shipped `localrag/mcp/` as a hand-rolled JSON-RPC dispatcher (`McpServer`) over a hand-written `TOOLS` schema list, a single FastAPI `POST /mcp` route, and a manual stdin loop.
`docs/mcp.md` documented this as a deliberate choice: "MCP is intentionally implemented without adding an SDK dependency."

That choice had a real cost. The hand-rolled adapter implemented only `initialize`, `tools/list`, and `tools/call` — a subset of the MCP specification — and every schema change had to be kept in sync by hand between the Python function signature and the `TOOLS` JSON Schema literal. Standard MCP clients and tools (MCP Inspector, off-the-shelf client libraries) expect the full protocol: notifications, the streamable-HTTP session lifecycle, and schema-generation conventions the hand-rolled adapter did not implement.

FastMCP (`fastmcp` on PyPI) is the de facto community SDK for building MCP servers in Python and is developed in coordination with the official `mcp` SDK. The published PyPI release is **3.4.6** at the time of this decision — the project's own docs site (gofastmcp.com) describes an in-development v4, but that version is not yet on PyPI, so the dependency is pinned `fastmcp>=3.4,<4` to track what actually installs rather than the aspirational docs.

## Decision

`localrag/mcp/server.py` now builds a `FastMCP` server via a module-level factory, `build_mcp_server(settings, engine_factory=None, ingestion_service_factory=None, collection_repo_factory=None, lifespan=None) -> FastMCP`.
It registers the same four tools as before — `search_documents`, `answer_question`, `ingest_path`, `list_collections` — with the same argument names, so existing callers are unaffected.
Each tool body still delegates to `localrag.application.service` exactly as before; `ApplicationError` and `(KeyError, TypeError, ValueError)` map to `fastmcp.exceptions.ToolError`, replacing the hand-written `{"isError": true, ...}` envelope.

Dependencies (`engine`, `ingestion_service`, `collection_repo`) are injected as **factories**, not instances. FastMCP registers tools once at server-construction time, so a factory lets `localrag/mcp/app.py` pass the existing `lru_cache`d `application.container` getters directly: the embedder, vector store, and retriever are still not built until the first real tool call, matching the previous adapter's behavior of only loading dependencies for `tools/call`, not `tools/list`/`initialize`.

Static API-key authentication does not fit FastMCP's built-in `auth=` providers, which target JWT/OAuth flows. Instead, `ApiKeyMiddleware` (a `fastmcp.server.middleware.Middleware` subclass) checks the key on the `on_request` hook, which gates every method — `initialize` and `tools/list` as well as `tools/call`. Gating only `tools/call` would have been a regression: the hand-rolled adapter ADR 038 describes rejected unauthenticated callers on every method, and the tool list itself discloses deployment surface.

Over HTTP the middleware reads `X-API-Key` via `fastmcp.server.dependencies.get_http_headers()`; over stdio, where no HTTP headers exist, it falls back to the `MCP_API_KEY` environment variable — the same fallback contract ADR 038 established. Because the check covers `initialize`, a rejected key fails the session handshake and surfaces as a protocol-level JSON-RPC `error` object carrying `"Invalid or missing API key."`, rather than as a tool-level `isError` result. This was verified against both an in-process `fastmcp.Client` and a live `uvicorn` process, not assumed from documentation.

`localrag/mcp/app.py` replaces the FastAPI route with `build_mcp_server(...).http_app(path="/mcp", stateless_http=True)`. `FastMCP.http_app()` has no `lifespan=` parameter of its own; the settings-loading/dependency-teardown lifespan is instead passed into the `FastMCP` constructor via `build_mcp_server(..., lifespan=lifespan)`, which `http_app()`'s returned ASGI app does run (confirmed by observing the lifespan's startup/shutdown log lines around a live `POST /mcp` request). `stateless_http=True` was chosen because the Compose deployment and any curl-based operator workflow send one-shot requests; without it, the streamable-HTTP transport requires a session-negotiation handshake that a stateless `curl` call cannot satisfy.

`localrag/mcp/__main__.py` now calls `mcp.run(transport="stdio")` after loading settings the same way the HTTP path does. FastMCP's stdio transport writes its startup banner and logs to stderr, leaving stdout clean for protocol JSON — verified by running `uv run python -m localrag.mcp` and inspecting stdout/stderr separately.

## Consequences

- `fastmcp>=3.4,<4` becomes a core dependency (not optional), because `localrag/mcp/` ships in the wheel per ADR 038's package map.
- The MCP wire surface is now the full MCP specification implemented by FastMCP/the underlying `mcp` SDK, not the three-method subset ADR 038 described. Existing tool names, arguments, and the `INGEST_ROOTS` / API-key contracts are unchanged.
- `docs/mcp.md`'s claim that MCP avoids an SDK dependency is removed; the curl example now sends the `Accept: application/json, text/event-stream` header the streamable-HTTP transport requires, and was run against a live container as part of writing this ADR.
- New transports or tool additions build on `fastmcp.FastMCP`/`fastmcp.tools` idioms (type hints, docstrings, `Annotated[..., Field(...)]`) instead of hand-written JSON Schema, and continue to depend on `localrag.application`, not `localrag.api`, per ADR 038.

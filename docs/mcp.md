# MCP

LocalRAG exposes the same application use cases through the Model Context Protocol.
The MCP adapter is a transport boundary; ingestion, retrieval, path policy, and collection access remain in `localrag/application/`.
The adapter is built on the [FastMCP](https://gofastmcp.com) SDK (`fastmcp` on PyPI, pinned `>=3.4,<4`).

## Tools

| Tool | Purpose |
| --- | --- |
| `search_documents` | Retrieve document chunks without generation |
| `answer_question` | Generate an answer with source citations |
| `ingest_path` | Ingest an allowed file or directory |
| `list_collections` | List available collections |

`ingest_path` applies `INGEST_ROOTS` exactly like the HTTP ingest use cases.
The tool does not provide a way to bypass configured roots.

## HTTP

The Compose stack runs MCP separately from the REST API at `http://127.0.0.1:8002/mcp`.
The endpoint uses FastMCP's streamable-HTTP transport (`stateless_http=True`), so every request is a self-contained `POST` — no session negotiation is required for a single call.
Send an `Accept` header that includes both `application/json` and `text/event-stream`; the server replies with a single Server-Sent Events message containing the JSON-RPC response.
Provide `X-API-Key` when `API_KEY` is configured.

```bash
curl -s http://127.0.0.1:8002/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'X-API-Key: change-me' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

This curl command was run against a live `localrag-mcp` container as part of verifying this doc; the response is an `event: message` / `data: {...}` block containing the four tools listed above.

The API and MCP containers share the `localrag_data` volume, but have independent processes and entrypoints.

## Stdio

For local MCP clients, run:

```bash
uv run python -m localrag.mcp
```

This starts FastMCP's stdio transport. Protocol messages follow the standard MCP JSON-RPC handshake (`initialize`, then `notifications/initialized`, then normal requests), one JSON object per line on stdin/stdout.
Startup banners and logs go to stderr, so stdout only ever carries protocol JSON.
When `API_KEY` is configured, set `MCP_API_KEY` for the stdio process — stdio has no HTTP headers, so the API key check falls back to this environment variable.

## Auth

Both transports enforce the same API key contract as the REST API, implemented as a FastMCP `Middleware` (`ApiKeyMiddleware` in `localrag/mcp/server.py`) rather than one of FastMCP's built-in JWT/OAuth `auth=` providers, since LocalRAG uses a single static key, not per-user tokens.
HTTP reads `X-API-Key` via `fastmcp.server.dependencies.get_http_headers()`; stdio reads `MCP_API_KEY` from the environment because no HTTP headers exist there.
The check runs on `on_request`, so it gates every method — `initialize` and `tools/list` as well as `tools/call`. This matches the hand-rolled adapter it replaced, which rejected unauthenticated callers on every method, and keeps the tool list from disclosing deployment surface to an unauthenticated caller.

A mismatched or missing key is therefore rejected at the protocol level, not as a tool result: the JSON-RPC response carries an `error` object with the message `"Invalid or missing API key."`. Because `initialize` is gated too, an unauthenticated MCP client fails while opening the session rather than on its first tool call.

## Wire surface

FastMCP implements the full MCP specification (`initialize`, `tools/list`, `tools/call`, notifications, and the streamable-HTTP session lifecycle) rather than the minimal hand-rolled subset LocalRAG previously exposed.
Client libraries and MCP Inspector work against this server without a bespoke client.

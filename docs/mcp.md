# MCP

LocalRAG exposes the same application use cases through the Model Context Protocol.
The MCP adapter is a transport boundary; ingestion, retrieval, path policy, and collection access remain in `localrag/application/`.

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
Send JSON-RPC MCP messages with `POST` and provide `X-API-Key` when `API_KEY` is configured.

```bash
curl -s http://127.0.0.1:8002/mcp \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: change-me' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

The API and MCP containers share the `localrag_data` volume, but have independent processes and entrypoints.

## Stdio

For local MCP clients, run:

```bash
uv run python -m localrag.mcp
```

When `API_KEY` is configured, set `MCP_API_KEY` for the stdio process.
Protocol messages use one JSON object per input line and one response object per output line.

MCP is intentionally implemented without adding an SDK dependency to the default installation.
The supported wire surface is `initialize`, `tools/list`, and `tools/call`.

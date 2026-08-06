from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from localrag.application import service as application_service
from localrag.application.dto import IngestDirectoryRequest, IngestFileRequest, QueryRequest
from localrag.application.errors import ApplicationError
from localrag.application.repository import ChromaCollectionRepository
from localrag.ingestion.service import IngestionService
from localrag.rag.engine import RAGEngine
from localrag.settings import Settings

PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": "Retrieve relevant document chunks without generating an answer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to search for."},
                "n_results": {"type": "integer", "minimum": 1},
                "metadata_filter": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
    {
        "name": "answer_question",
        "description": "Answer a question from the ingested document collection with citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "model": {"type": "string"},
                "n_results": {"type": "integer", "minimum": 1},
                "metadata_filter": {"type": "object", "additionalProperties": {"type": "string"}},
            },
            "required": ["question"],
        },
    },
    {
        "name": "ingest_path",
        "description": "Ingest an allowed file or directory into the document collection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "embed_model": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_collections",
        "description": "List available document collections.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class McpServer:
    """Small MCP JSON-RPC server shared by stdio and HTTP transports."""

    def __init__(
        self,
        settings: Settings,
        engine: RAGEngine | None = None,
        ingestion_service: IngestionService | None = None,
        collection_repo: ChromaCollectionRepository | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.ingestion_service = ingestion_service
        self.collection_repo = collection_repo

    def handle_message(
        self, message: dict[str, Any], api_key: str | None = None
    ) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None
        if self.settings.api_key and api_key != self.settings.api_key:
            return self._error(request_id, -32001, "Invalid or missing API key.")
        if method == "initialize":
            return self._response(
                request_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "localrag", "version": "0.1.0"},
                },
            )
        if method == "tools/list":
            return self._response(request_id, {"tools": TOOLS})
        if method == "tools/call":
            return self._response(request_id, self._call_tool(message, api_key))
        return self._error(request_id, -32601, f"Unknown method: {method}")

    def _call_tool(self, message: dict[str, Any], api_key: str | None) -> dict[str, Any]:
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            result = self._dispatch(str(name), arguments)
        except ApplicationError as exc:
            return self._tool_error(exc.detail)
        except (KeyError, TypeError, ValueError) as exc:
            return self._tool_error(str(exc))
        return {"content": [{"type": "text", "text": json.dumps(result, default=str)}]}

    def _dispatch(self, name: str, arguments: dict[str, Any]) -> object:
        if name in {"search_documents", "answer_question"}:
            request = QueryRequest(
                question=str(arguments["question"]),
                model=arguments.get("model"),
                n_results=arguments.get("n_results"),
                metadata_filter=arguments.get("metadata_filter"),
            )
        if name == "search_documents":
            return application_service.get_query_contexts(request, self._engine())
        if name == "answer_question":
            return asdict(application_service.query_json(request, self._engine()))
        if name == "ingest_path":
            path = str(arguments["path"])
            if Path(path).is_dir():
                result = application_service.ingest_directory(
                    IngestDirectoryRequest(
                        path=path,
                        recursive=arguments.get("recursive"),
                        embed_model=arguments.get("embed_model"),
                    ),
                    self.settings,
                    self._ingestion_service(),
                )
            else:
                return asdict(
                    application_service.ingest_file(
                        IngestFileRequest(path=path, embed_model=arguments.get("embed_model")),
                        self.settings,
                        self._ingestion_service(),
                    )
                )
            return asdict(result)
        if name == "list_collections":
            return asdict(application_service.list_collections_response(self._collection_repo()))
        message = f"Unknown tool: {name}"
        raise ValueError(message)

    def _engine(self) -> RAGEngine:
        if self.engine is None:
            raise RuntimeError("MCP engine is not configured.")
        return self.engine

    def _ingestion_service(self) -> IngestionService:
        if self.ingestion_service is None:
            raise RuntimeError("MCP ingestion service is not configured.")
        return self.ingestion_service

    def _collection_repo(self) -> ChromaCollectionRepository:
        if self.collection_repo is None:
            raise RuntimeError("MCP collection repository is not configured.")
        return self.collection_repo

    @staticmethod
    def _response(request_id: object, result: object) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: object, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {"isError": True, "content": [{"type": "text", "text": message}]}

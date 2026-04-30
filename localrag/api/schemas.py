from __future__ import annotations

from typing import Annotated

from fastapi import Path
from pydantic import BaseModel, ConfigDict, Field

from localrag.settings import DEFAULT_OLLAMA_EMBED_MODEL, DEFAULT_OLLAMA_LLM_MODEL

CollectionNamePath = Annotated[
    str,
    Path(
        description=("Chroma collection name to remove (vectors and metadata for that namespace)."),
        examples=["localrag"],
    ),
]


class QueryRequest(BaseModel):
    """Body for ``POST /query``.

    Optional fields use server ``Settings`` / ``.env`` when omitted.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "How do I run the LocalRAG API?",
                    "model": DEFAULT_OLLAMA_LLM_MODEL,
                    "n_results": 5,
                },
                {
                    "question": "What is chunk overlap?",
                },
            ]
        }
    )

    question: str = Field(
        description="User question answered from ingested document chunks (RAG).",
        examples=["What does the README say about installation?"],
    )
    model: str | None = Field(
        default=None,
        description=(
            "Ollama **chat** model tag (`ollama list`), e.g. the LLM you pulled with your "
            "embedding model. If omitted, uses the server's configured LLM (default tag "
            f"`{DEFAULT_OLLAMA_LLM_MODEL}` via `OLLAMA_LLM_MODEL`)."
        ),
        examples=[DEFAULT_OLLAMA_LLM_MODEL],
    )
    n_results: int | None = Field(
        default=None,
        description=(
            "How many chunks to retrieve from the vector store before generating an answer. "
            "If omitted, uses the server's `rag_top_k` setting."
        ),
        examples=[5],
    )


class IngestFileRequest(BaseModel):
    """Body for ``POST /ingest``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"path": "/home/user/project/README.md"},
                {"path": "C:\\\\Users\\\\you\\\\Documents\\\\notes.txt"},
            ]
        }
    )

    path: str = Field(
        description=(
            "Path to an existing **file** on the machine running the API "
            "(LocalRAG is local-first). "
            "URL-encoded segments (e.g. `%20` for spaces) are decoded before use."
        ),
        examples=["/var/docs/guide.md", "C:\\\\docs\\\\report.txt"],
    )
    embed_model: str | None = Field(
        default=None,
        description=(
            "Ollama **embedding** model tag for this ingest run (`ollama list`). "
            "If omitted, uses the server's configured embed model "
            f"(`{DEFAULT_OLLAMA_EMBED_MODEL}` via `OLLAMA_EMBED_MODEL`)."
        ),
        examples=[DEFAULT_OLLAMA_EMBED_MODEL],
    )


class IngestDirectoryRequest(BaseModel):
    """Body for ``POST /ingest/directory``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"path": "/home/user/project/docs", "recursive": True},
                {"path": "C:\\\\Users\\\\you\\\\notes"},
            ]
        }
    )

    path: str = Field(
        description=(
            "Path to an existing **directory** on the server. "
            "URL-encoded segments (e.g. `%20`) are decoded before use."
        ),
        examples=["/var/docs", "C:\\\\project\\\\src"],
    )
    recursive: bool | None = Field(
        default=None,
        description=(
            "Whether to walk subdirectories. If omitted, uses the server's "
            "`ingest_recursive` setting."
        ),
        examples=[True],
    )
    embed_model: str | None = Field(
        default=None,
        description=(
            "Ollama **embedding** model tag for this ingest run (`ollama list`). "
            "If omitted, uses the server's configured embed model "
            f"(`{DEFAULT_OLLAMA_EMBED_MODEL}` via `OLLAMA_EMBED_MODEL`)."
        ),
        examples=[DEFAULT_OLLAMA_EMBED_MODEL],
    )


class IngestFileResponse(BaseModel):
    status: str = Field(
        description="Literal status for a successful ingest.",
        examples=["ok"],
    )
    chunks_added: int = Field(
        description="Number of text chunks written to the vector store for this file.",
        examples=[12],
    )
    source: str = Field(
        description="Resolved source identifier (usually the file path) stored on chunk metadata.",
        examples=["/home/user/README.md"],
    )


class IngestDirectoryResponse(BaseModel):
    status: str = Field(
        description="Literal status for a successful directory ingest.",
        examples=["ok"],
    )
    files_processed: int = Field(
        description="Number of files ingested under the directory (after filters).",
        examples=[8],
    )
    total_chunks: int = Field(
        description="Total chunks added across all files in this run.",
        examples=[42],
    )


class HealthResponse(BaseModel):
    status: str = Field(
        description="Overall API process status.",
        examples=["ok"],
    )
    ollama_ok: bool = Field(
        description=(
            "Whether the Ollama HTTP API responded successfully to `GET /api/tags` "
            "within the health timeout."
        ),
        examples=[True],
    )
    chroma_path: str = Field(
        description="Configured Chroma persist directory (`CHROMA_PERSIST_PATH`).",
        examples=["./data/chroma"],
    )
    collections: list[str] = Field(
        description=(
            "Collection names reported by the vector store (same process as ingest/query)."
        ),
        examples=[["localrag"]],
    )


class SourceRef(BaseModel):
    source: str = Field(
        description="File path or identifier of the source chunk.", examples=["/docs/guide.md"]
    )
    chunk_index: int = Field(
        description="Zero-based index of the chunk within the source.", examples=[0]
    )


class QueryResponse(BaseModel):
    """Response body for ``POST /query`` (JSON mode)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "You can run the API with `uv run uvicorn localrag.api.main:app`.",
                    "sources": [{"source": "/docs/README.md", "chunk_index": 2}],
                    "latency_ms": 312.5,
                    "model": "llama3.2",
                }
            ]
        }
    )

    answer: str = Field(
        description="Generated answer text.",
        examples=["The README describes installation steps."],
    )
    sources: list[SourceRef] = Field(description="Source chunks used to generate the answer.")
    latency_ms: float = Field(
        description="Total wall-clock latency in milliseconds.", examples=[312.5]
    )
    model: str = Field(description="Model tag used to generate the answer.", examples=["llama3.2"])


class AgentQueryRequest(BaseModel):
    """Body for ``POST /agent/query``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"question": "What does the README say about installation?"}]
        }
    )

    question: str = Field(
        description="User question for the agent to handle.",
        examples=["What does the README say about installation?"],
    )
    model: str = Field(
        default="claude-haiku-4-5",
        description="Anthropic model tag to use for tool-use decisions.",
        examples=["claude-haiku-4-5"],
    )


class AgentQueryResponse(BaseModel):
    """Response body for ``POST /agent/query``."""

    answer: str = Field(description="Generated answer.", examples=["Installation is described..."])
    tool_used: str = Field(
        description="Tool the agent chose: search_documents or answer_directly.",
        examples=["search_documents"],
    )
    reasoning: str = Field(
        description="Agent reasoning for the tool choice.",
        examples=["Used search_documents with query: 'installation'"],
    )
    sources: list[SourceRef] = Field(
        description="Source chunks used (empty when tool_used=answer_directly).",
        default_factory=list,
    )
    latency_ms: float = Field(description="Total latency in milliseconds.", examples=[450.0])
    model: str = Field(description="Model used for agent decisions.", examples=["claude-haiku-4-5"])


class CollectionListResponse(BaseModel):
    collections: list[str] = Field(
        description="Names of Chroma collections on the configured persist path.",
        examples=[["localrag", "experiments"]],
    )


class CollectionDeleteResponse(BaseModel):
    status: str = Field(
        description="Literal status after the collection was deleted.",
        examples=["ok"],
    )


class RebuildCollectionRequest(BaseModel):
    """Body for ``POST /collections/rebuild`` (may be empty ``{}``)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {},
                {"embed_model": DEFAULT_OLLAMA_EMBED_MODEL},
            ]
        }
    )

    embed_model: str | None = Field(
        default=None,
        description=(
            "Ollama **embedding** model tag used when re-embedding all stored sources. "
            "If omitted, uses the server's configured embed model "
            f"(`{DEFAULT_OLLAMA_EMBED_MODEL}` via `OLLAMA_EMBED_MODEL`)."
        ),
        examples=[DEFAULT_OLLAMA_EMBED_MODEL],
    )


class RebuildCollectionResponse(BaseModel):
    status: str = Field(
        description="Literal status after rebuild.",
        examples=["ok"],
    )
    files_processed: int = Field(
        description="Files successfully re-ingested.",
        examples=[3],
    )
    total_chunks: int = Field(
        description="Total chunks written after rebuild.",
        examples=[40],
    )
    missing_sources: list[str] = Field(
        description=(
            "Source paths that were in the vector store but no longer exist on disk; "
            "their vectors were removed."
        ),
        examples=[["/old/path/removed.md"]],
    )

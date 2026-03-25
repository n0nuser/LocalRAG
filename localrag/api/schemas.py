from __future__ import annotations

from typing import Annotated

from fastapi import Path
from pydantic import BaseModel, ConfigDict, Field

from localrag.settings import DEFAULT_OLLAMA_LLM_MODEL

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

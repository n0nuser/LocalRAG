"""Environment-backed settings (``.env`` + process env).

Use :func:`get_settings` for a cached singleton. Variable names match
:class:`Settings` fields (case-insensitive), e.g. ``OLLAMA_BASE_URL``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Defaults for Ollama model tags (`ollama pull` / `ollama list`).
# Keep in sync with docs and API examples.
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.2"


class Settings(BaseSettings):
    """Application configuration loaded from the environment and optional ``.env``.

    **Ollama** — ``ollama_base_url`` is the HTTP API root (embeddings and chat).
    ``ollama_embed_model`` / ``ollama_llm_model`` are model tags as shown by
    ``ollama list``.

    **Chroma** — ``chroma_persist_path`` is the on-disk store directory;
    ``chroma_collection_name`` namespaces vectors for this app instance.

    **Ingestion** — Text is split into chunks of up to ``chunk_chars`` characters
    with ``chunk_overlap_chars`` shared between neighbors. Embeddings are sent
    to Ollama in batches of ``embedding_batch_size``. Directory ingest uses
    ``ingest_recursive`` when not overridden per request. If ``ingest_roots`` is
    non-empty, only files and directories under those paths (after resolving) are
    allowed through the HTTP ingest API; an empty list disables that restriction.

    **RAG** — ``rag_top_k`` is how many chunks are retrieved for context.
    ``rag_system_prompt`` is the system message for the answering model.

    **API** — ``api_host`` / ``api_port`` are the uvicorn bind address and port.

    **Logging** — ``log_level`` is the minimum level for the ``localrag`` logger
    (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``). Used when the API starts and
    when the CLI process starts.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL
    ollama_llm_model: str = DEFAULT_OLLAMA_LLM_MODEL

    chroma_persist_path: str = "./data/chroma"
    chroma_collection_name: str = "localrag"

    chunk_chars: int = 512
    chunk_overlap_chars: int = 50
    embedding_batch_size: int = 32

    ingest_recursive: bool = True
    ingest_roots: list[str] = []

    rag_top_k: int = 5
    rag_system_prompt: str = (
        "You are a helpful assistant. Answer only based on the provided context."
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def is_path_allowed(candidate: Path, roots: list[str]) -> bool:
    """Return whether ``candidate`` may be ingested when ``roots`` is restricted.

    If ``roots`` is empty, every path is allowed. Otherwise ``candidate`` must be
    the same as, or nested under, at least one resolved root path.
    """
    if not roots:
        return True

    resolved_candidate = candidate.resolve()
    for root in roots:
        resolved_root = Path(root).resolve()
        if resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents:
            return True
    return False

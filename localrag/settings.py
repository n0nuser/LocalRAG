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
DEFAULT_OLLAMA_LLM_MODEL = "gemma3:4b"


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
    ``POST /ingest/upload`` bypasses ``ingest_roots`` (the server chooses the
    destination) but enforces ``upload_max_bytes`` and saves under ``upload_dir``.

    **PDF OCR** — When ``ocr_enabled`` is true (default), PDF pages whose extracted
    text layer is shorter than ``ocr_min_chars_per_page`` (scanned/image-only pages)
    are rasterized and run through Tesseract OCR (``ocr_language`` is a Tesseract
    language code, e.g. ``eng``). Requires the ``tesseract`` binary on the host; if
    it is missing, OCR fails silently per-page and the original (possibly empty)
    text-layer output is kept. See `docs/ocr.md`.

    **RAG** — ``rag_top_k`` is how many chunks are retrieved for context.
    ``rag_system_prompt`` is the system message for the answering model.
    When ``parent_expansion_enabled`` is true (default), top hits with a
    non-empty ``heading_path`` are expanded to their full sibling-chunk
    section before prompting; set false to disable. ``rag_min_context_score``
    gates generation on retrieval confidence: below this score (or with no
    contexts at all) the engine returns a canned refusal instead of calling
    the LLM; ``0`` (default) disables the gate.

    **Reranking** — When ``rerank_enabled`` is true (default false, requires
    ``uv sync --extra rerank``), retrieval over-fetches ``rerank_fetch_k``
    candidates and a local cross-encoder (``rerank_model``) re-scores and
    trims them to ``rag_top_k`` before freshness/expansion.

    **Query rewriting** — When ``query_rewrite_enabled`` is true (default
    false), an extra LLM round-trip rewrites the question into a keyword-dense
    search query before embedding/BM25 retrieval; the original question is
    still used for the final answer prompt.

    **API** — ``api_host`` / ``api_port`` are the uvicorn bind address and port.

    **Logging** — ``log_level`` is the minimum level for the ``localrag`` logger
    (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``). Used when the API starts and
    when the CLI process starts.

    **Tenant tagging** — ``tenant_id`` (empty by default) is written to every
    chunk's metadata at ingest time and can be used as a
    ``QueryRequest.metadata_filter`` key (``{"tenant_id": "..."}``) to scope
    retrieval to one tenant. See `docs/rag-retrieval.md`.
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
    chunk_overlap_chars: int = 150
    chunking_mode: str = "structural"
    chunk_max_chars: int = 1200
    chunk_min_chars: int = 200
    embedding_batch_size: int = 32

    ingest_recursive: bool = True
    ingest_roots: list[str] = []

    upload_dir: str = "./data/uploads"
    upload_max_bytes: int = 100_000_000

    audit_log_path: str = ""

    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_min_chars_per_page: int = 20

    rag_top_k: int = 5
    rag_min_context_score: float = 0.0
    retrieval_mode: str = "hybrid"
    bm25_weight: float = 0.5
    rrf_k: int = 60
    freshness_half_life_days: float = 30.0
    parent_expansion_enabled: bool = True
    query_rewrite_enabled: bool = False
    rag_system_prompt: str = (
        "You are a helpful assistant. Answer only based on the provided context."
    )

    # Optional cross-encoder reranking (requires `uv sync --extra rerank`).
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20

    api_host: str = "0.0.0.0"  # nosec B104 — configurable bind address, default intentional
    api_port: int = 8000

    # Optional API key enforced on all non-health endpoints via X-API-Key header.
    # Leave empty (default) to disable authentication.
    api_key: str = ""

    # LLM backend selector: "ollama" | "openai" | "anthropic"
    llm_backend: str = "ollama"

    # Canonical embedding model alias (maps to ollama_embed_model when backend=ollama).
    embedding_model: str = ""

    # OpenAI provider settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Anthropic provider settings
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    # Agent settings (uses Anthropic tool-use)
    agent_model: str = "claude-haiku-4-5"

    # Optional automatic failover backend when the primary trips its circuit breaker
    # ("ollama" | "openai" | "anthropic"); empty disables fallback.
    llm_fallback_backend: str = ""
    llm_retry_max_attempts: int = 3
    llm_circuit_fail_max: int = 5
    llm_circuit_reset_timeout_seconds: float = 30.0

    log_level: str = "INFO"

    # Optional tenant tag written to every chunk's metadata and usable as a
    # QueryRequest.metadata_filter key ({"tenant_id": "..."}). Empty = untagged
    # (single-tenant deployments, the common case, pay zero extra cost).
    tenant_id: str = ""


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

"""Environment-backed settings (``.env`` + process env).

Use :func:`get_settings` for a cached singleton. Variable names match
:class:`Settings` fields (case-insensitive), e.g. ``OLLAMA_BASE_URL``.
"""

from __future__ import annotations

import contextvars
import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource

# Defaults for Ollama model tags (`ollama pull` / `ollama list`).
# Keep in sync with docs and API examples.
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_LLM_MODEL = "gemma3:4b"

_yaml_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "localrag_yaml_path", default=None
)
_active_settings: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "localrag_active_settings", default=None
)
_SECRET_FIELDS = {"api_key", "openai_api_key", "anthropic_api_key"}
_PATH_FIELDS = {"chroma_persist_path", "upload_dir", "audit_log_path", "ingest_roots"}
_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when an explicitly selected configuration file cannot be loaded."""


class _YamlSource(PydanticBaseSettingsSource):
    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        path = _yaml_path.get()
        if path is None:
            return {}
        return _read_yaml(path)


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        return _INTERPOLATION.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)), value
        )
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:  # noqa: C901, PLR0912
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")  # noqa: EM102
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Unable to parse YAML configuration {path}: {exc}") from exc  # noqa: EM102
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigError("YAML configuration must contain a mapping at its root")

    sections = {
        "embedding": {
            "provider": "embedding_provider",
            "model": "embedding_model",
            "timeout_seconds": "embedding_timeout_seconds",
            "batch_size": "embedding_batch_size",
            "sentence_transformers_model": "sentence_transformers_model",
        },
        "retrieval": {
            "top_k": "rag_top_k",
            "min_context_score": "rag_min_context_score",
            "mode": "retrieval_mode",
            "bm25_weight": "bm25_weight",
            "rrf_k": "rrf_k",
            "freshness_half_life_days": "freshness_half_life_days",
            "freshness_weight": "freshness_weight",
            "parent_expansion_enabled": "parent_expansion_enabled",
            "query_rewrite_enabled": "query_rewrite_enabled",
            "rerank_enabled": "rerank_enabled",
            "rerank_model": "rerank_model",
            "rerank_fetch_k": "rerank_fetch_k",
        },
        "generation": {
            "backend": "llm_backend",
            "ollama_model": "ollama_llm_model",
            "temperature": "llm_temperature",
            "seed": "llm_seed",
            "system_prompt": "rag_system_prompt",
            "fallback_backend": "llm_fallback_backend",
            "retry_max_attempts": "llm_retry_max_attempts",
            "circuit_fail_max": "llm_circuit_fail_max",
            "circuit_reset_timeout_seconds": "llm_circuit_reset_timeout_seconds",
        },
        "dataset": {"roots": "ingest_roots", "recursive": "ingest_recursive"},
        "evaluation": {"seed": "eval_seed"},
    }
    fields = set(Settings.model_fields)
    flattened: dict[str, Any] = {}
    for key, value in document.items():
        if key in sections:
            if not isinstance(value, dict):
                raise ConfigError(f"YAML section {key} must be a mapping")  # noqa: EM102
            for nested_key, nested_value in value.items():
                target = sections[key].get(nested_key)
                if target is None:
                    raise ConfigError(f"Unknown YAML key: {key}.{nested_key}")  # noqa: EM102
                flattened[target] = nested_value
        elif key == "ollama":
            if not isinstance(value, dict) or set(value) - {"embed_model", "base_url"}:
                unknown = next(iter(set(value) - {"embed_model", "base_url"}), key)
                raise ConfigError(f"Unknown YAML key: ollama.{unknown}")  # noqa: EM102
            warnings.warn(
                "ollama.embed_model and ollama.base_url YAML keys are deprecated; "
                "use top-level legacy field names or embedding.*. "
                "Removal is planned after the next major release.",
                DeprecationWarning,
                stacklevel=3,
            )
            if "embed_model" in value:
                flattened["ollama_embed_model"] = value["embed_model"]
            if "base_url" in value:
                flattened["ollama_base_url"] = value["base_url"]
        elif key in fields:
            if key in _SECRET_FIELDS and value and not (isinstance(value, str) and "${" in value):
                raise ConfigError("Secrets must be supplied through the environment")
            flattened[key] = value
        else:
            raise ConfigError(f"Unknown YAML key: {key}")  # noqa: EM102

    flattened = _interpolate(flattened)
    base = path.parent
    for field in ("chroma_persist_path", "upload_dir", "audit_log_path"):
        if flattened.get(field):
            flattened[field] = str(_resolve_path(str(flattened[field]), base))
    if flattened.get("ingest_roots"):
        flattened["ingest_roots"] = [
            str(_resolve_path(str(root), base)) for root in flattened["ingest_roots"]
        ]
    return flattened


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


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

    **PDF OCR** — When ``ocr_enabled`` is true (default), PDF pages that ``pdf-inspector``
    flags as unreliable, or whose extracted Markdown is shorter than
    ``ocr_min_chars_per_page`` (scanned/image-only pages), are rasterized and run
    through Tesseract OCR (``ocr_language`` is a Tesseract language code, e.g.
    ``eng``). Requires the ``tesseract`` binary on the host; if it is missing, OCR
    fails silently per-page and the original (possibly empty) Markdown output is
    kept. See `docs/ocr.md`.

    **RAG** — ``rag_top_k`` is how many chunks are retrieved for context.
    ``rag_system_prompt`` is the system message for the answering model.
    When ``parent_expansion_enabled`` is true (default), top hits with a
    non-empty ``heading_path`` are expanded to their full sibling-chunk
    section before prompting; set false to disable. ``rag_min_context_score``
    gates generation on retrieval confidence: below this score (or with no
    contexts at all) the engine returns a canned refusal instead of calling
    the LLM; ``0`` (default) disables the gate.

    **Recency** — In hybrid mode, recency joins RRF as its own ranked list
    weighted by ``freshness_weight`` (taken out of the relevance budget), so it
    breaks near-ties without overturning a clearly better match; ties on
    relevance are resolved newest-first, and chunks with no usable
    ``ingested_at`` take the middle recency rank. In vector-only mode the
    multiplicative ``0.5 ** (age_days / freshness_half_life_days)`` decay still
    applies. Either ``freshness_half_life_days=0`` or ``freshness_weight=0``
    disables recency. See `docs/adr/006-freshness-decay.md`.

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _YamlSource(settings_cls),
            file_secret_settings,
        )

    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL
    ollama_llm_model: str = DEFAULT_OLLAMA_LLM_MODEL

    # Generation sampling. Both default to unset so Ollama keeps its per-model
    # defaults; set them (e.g. 0.0 / any int) to make answers reproducible.
    llm_temperature: float | None = None
    llm_seed: int | None = None

    chroma_persist_path: str = "./data/chroma"
    chroma_collection_name: str = "localrag"

    chunk_chars: int = 512
    chunk_overlap_chars: int = 150
    chunking_mode: str = "structural"
    chunk_max_chars: int = 1200
    chunk_min_chars: int = 200
    embedding_batch_size: int = 32
    embedding_provider: str = "ollama"
    embedding_timeout_seconds: float = 120.0
    sentence_transformers_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ingest_recursive: bool = True
    ingest_roots: list[str] = []

    # Caps concurrently pending/running async ingest jobs; further submissions get 429.
    max_pending_ingest_jobs: int = 10

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
    freshness_weight: float = 0.15
    parent_expansion_enabled: bool = True
    query_rewrite_enabled: bool = False
    rag_system_prompt: str = (
        "You are a helpful assistant. Answer only based on the provided context."
    )

    # Optional cross-encoder reranking (requires `uv sync --extra rerank`).
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_fetch_k: int = 20

    # In-process TTL cache for repeated/near-identical queries (0 disables; no external cache).
    query_cache_ttl_seconds: float = 0.0
    query_cache_maxsize: int = 256

    api_host: str = "0.0.0.0"  # nosec B104 — configurable bind address, default intentional
    api_port: int = 8000

    # Optional API key enforced on all non-health endpoints via X-API-Key header.
    # Leave empty (default) to disable authentication.
    api_key: str = ""

    # LLM backend selector: "ollama" | "openai" | "anthropic"
    llm_backend: str = "ollama"

    # Canonical embedding model alias (maps to ollama_embed_model when backend=ollama).
    embedding_model: str = ""

    @property
    def effective_embedding_model(self) -> str:
        """Resolve the new model name while preserving legacy Ollama deployments."""
        return self.embedding_model.strip() or self.ollama_embed_model

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

    eval_seed: int = 42

    @model_validator(mode="after")
    def validate_configuration(self) -> Settings:
        if self.chunk_min_chars > self.chunk_max_chars:
            raise ValueError("chunk_min_chars must be less than or equal to chunk_max_chars")
        if self.retrieval_mode not in {"hybrid", "vector"}:
            raise ValueError("retrieval_mode must be 'hybrid' or 'vector'")
        if not 0 <= self.bm25_weight <= 1:
            raise ValueError("bm25_weight must be between 0 and 1")
        return self

    def resolved_snapshot(self) -> dict[str, Any]:
        """Return deterministic, non-secret configuration provenance."""
        snapshot = self.model_dump(mode="json")
        for field in _SECRET_FIELDS:
            snapshot[field] = "<redacted>"
        for field in _PATH_FIELDS:
            if snapshot.get(field):
                snapshot[field] = (
                    "<path>" if field != "ingest_roots" else ["<path>"] * len(snapshot[field])
                )
        return snapshot


@lru_cache(maxsize=32)
def _cached_settings(config_path: str | None, overrides: tuple[tuple[str, Any], ...]) -> Settings:
    token = _yaml_path.set(Path(config_path).expanduser().resolve() if config_path else None)
    try:
        return Settings(**dict(overrides))
    finally:
        _yaml_path.reset(token)


def load_settings(
    config_path: Path | str | None = None, cli_overrides: dict[str, Any] | None = None
) -> Settings:
    """Resolve one immutable settings object using all supported sources."""
    path = Path(config_path).expanduser() if config_path is not None else None
    overrides = tuple(sorted((cli_overrides or {}).items()))
    unknown = set(dict(overrides)) - set(Settings.model_fields)
    if unknown:
        field = sorted(unknown)[0]
        raise ConfigError(f"Unknown CLI override: {field}")  # noqa: EM102
    return _cached_settings(str(path) if path else None, overrides)


@lru_cache(maxsize=1)
def _default_settings(config_path: str | None) -> Settings:
    return load_settings(config_path)


def get_settings() -> Settings:
    """Return the process settings, loading ``LOCALRAG_CONFIG`` when selected."""
    active = _active_settings.get()
    if active is not None:
        return active
    return _default_settings(os.environ.get("LOCALRAG_CONFIG"))


def set_current_settings(settings: Settings) -> None:
    """Select settings for the current CLI/API execution context."""
    _active_settings.set(settings)


def clear_current_settings() -> None:
    """Return resolution to the cached process settings."""
    _active_settings.set(None)


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

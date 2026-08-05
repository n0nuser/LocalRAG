"""Environment-backed settings (``.env`` + process env).

Use :func:`get_settings` for a cached singleton. The **flat**, documented
variable names remain the public contract (case-insensitive), e.g.
``OLLAMA_BASE_URL`` or ``HYDE_ENABLED``. Internally the fields are organised
into the grouped sub-models in :mod:`localrag.settings_groups`, each owning its
own validation; :mod:`localrag.settings_map` maps between the two. See
`docs/adr/037-grouped-configuration-model.md`.
"""

from __future__ import annotations

import contextvars
import json
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

from localrag import settings_groups as _groups
from localrag.settings_groups import (
    ApiSettings,
    AuditSettings,
    ChromaSettings,
    ChunkingSettings,
    EmbeddingSettings,
    IngestSettings,
    LlmSettings,
    ObservabilitySettings,
    OcrSettings,
    OllamaSettings,
    QueryCacheSettings,
    RetrievalSettings,
    UploadSettings,
)
from localrag.settings_map import FLAT_TO_PATH, UNGROUPED_FIELDS

# Public import site for the Ollama defaults (localrag.api.schemas imports these
# from here). settings_groups defines the same values as its field defaults.
DEFAULT_OLLAMA_EMBED_MODEL = _groups.DEFAULT_OLLAMA_EMBED_MODEL
DEFAULT_OLLAMA_LLM_MODEL = _groups.DEFAULT_OLLAMA_LLM_MODEL

_yaml_path: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "localrag_yaml_path", default=None
)
_active_settings: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "localrag_active_settings", default=None
)
_SECRET_FIELDS = {"api_key", "openai_api_key", "anthropic_api_key"}
_PATH_FIELDS = {
    "chroma_persist_path",
    "upload_dir",
    "audit_log_path",
    "ingest_roots",
    "embedding_cache_path",
}
_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Flags retired in ADR 036: both features are unconditional now. The names are
# still accepted (and ignored) so existing configs keep loading; they only warn.
# Values map "section.key" for YAML, bare names for flat fields.
_RETIRED_FLAGS = {
    "embedding.cache_enabled",
    "retrieval.context_compression_enabled",
    "embedding_cache_enabled",
    "context_compression_enabled",
}


def _warn_retired_flag(name: str) -> None:
    warnings.warn(
        f"{name} is retired and ignored: embedding caching and context compression "
        "are always applied. Remove it from your configuration; their budget and "
        "path settings still control behavior.",
        DeprecationWarning,
        stacklevel=4,
    )


class ConfigError(ValueError):
    """Raised when an explicitly selected configuration file cannot be loaded."""


def _redact_path(snapshot: dict[str, Any], path: str, placeholder: str) -> None:
    """Replace a dotted path's value in a nested snapshot, if it is set."""
    *sections, leaf = path.split(".")
    cursor: Any = snapshot
    for section in sections:
        if not isinstance(cursor, dict) or section not in cursor:
            return
        cursor = cursor[section]
    if not isinstance(cursor, dict) or not cursor.get(leaf):
        return
    value = cursor[leaf]
    cursor[leaf] = [placeholder] * len(value) if isinstance(value, list) else placeholder


def _nest(flat_values: dict[str, Any]) -> dict[str, Any]:
    """Regroup flat field names onto the grouped sub-model paths.

    Values for unmapped names (the ungrouped fields, or anything unknown) are
    passed through untouched so pydantic reports them as it normally would.
    """
    nested: dict[str, Any] = {}
    for name, value in flat_values.items():
        path = FLAT_TO_PATH.get(name)
        if path is None:
            nested[name] = value
            continue
        *sections, leaf = path.split(".")
        cursor = nested
        for section in sections:
            existing = cursor.get(section)
            if not isinstance(existing, dict):
                existing = {}
                cursor[section] = existing
            cursor = existing
        cursor[leaf] = value
    return nested


class _FlatEnvSource(PydanticBaseSettingsSource):
    """Map documented flat env names (``HYDE_ENABLED``) onto grouped paths.

    Nested models are not otherwise reachable by their flat names — pydantic
    would expect ``RETRIEVAL__HYDE__ENABLED`` — so without this source every
    documented environment variable would silently stop working.
    """

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return _nest(_flat_from_mapping(os.environ))


class _FlatDotenvSource(PydanticBaseSettingsSource):
    """The same flat-name regrouping for ``.env``, which env vars still outrank."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        env_file = self.config.get("env_file")
        if not env_file:
            return {}
        path = Path(str(env_file))
        if not path.exists():
            return {}
        values: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip("\"'")
        return _nest(_flat_from_mapping(values))


_COMPLEX_FLAT_FIELDS = frozenset({"ingest_roots"})


def _flat_from_mapping(source: Any) -> dict[str, Any]:
    """Pick the documented flat names out of an env-like mapping, case-insensitively."""
    found: dict[str, Any] = {}
    for flat in FLAT_TO_PATH:
        for candidate in (flat.upper(), flat.lower()):
            if candidate in source:
                raw = source[candidate]
                # Complex fields are JSON-encoded in the environment, matching
                # pydantic-settings' own behavior for list/dict-valued settings.
                if flat in _COMPLEX_FLAT_FIELDS and isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        message = f"{candidate} must be valid JSON: {exc}"
                        raise ConfigError(message) from exc
                found[flat] = raw
                break
    return found


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
            "cache_path": "embedding_cache_path",
            "cache_max_entries": "embedding_cache_max_entries",
            "cache_max_bytes": "embedding_cache_max_bytes",
            "cache_preprocessing_version": "embedding_cache_preprocessing_version",
            "cache_task_prefix": "embedding_cache_task_prefix",
        },
        "retrieval": {
            "plugin": "retriever_plugin",
            "top_k": "rag_top_k",
            "min_context_score": "rag_min_context_score",
            "mode": "retrieval_mode",
            "bm25_weight": "bm25_weight",
            "rrf_k": "rrf_k",
            "freshness_half_life_days": "freshness_half_life_days",
            "freshness_weight": "freshness_weight",
            "parent_expansion_enabled": "parent_expansion_enabled",
            "query_rewrite_enabled": "query_rewrite_enabled",
            "query_expansion_enabled": "query_expansion_enabled",
            "query_expansion_max_variants": "query_expansion_max_variants",
            "query_expansion_max_query_chars": "query_expansion_max_query_chars",
            "experiment_mode": "retrieval_experiment_mode",
            "hyde_enabled": "hyde_enabled",
            "hyde_model": "hyde_model",
            "hyde_timeout_seconds": "hyde_timeout_seconds",
            "hyde_input_max_chars": "hyde_input_max_chars",
            "hyde_output_max_chars": "hyde_output_max_chars",
            "hyde_output_max_tokens": "hyde_output_max_tokens",
            "hyde_lexical_input": "hyde_lexical_input",
            "hyde_log_content": "hyde_log_content",
            "candidate_budget": "query_expansion_candidate_budget",
            "rerank_enabled": "rerank_enabled",
            "rerank_model": "rerank_model",
            "rerank_fetch_k": "rerank_fetch_k",
            "context_compression_candidate_count": "context_compression_candidate_count",
            "context_compression_max_contexts": "context_compression_max_contexts",
            "context_compression_per_context_tokens": "context_compression_per_context_tokens",
            "context_compression_total_tokens": "context_compression_total_tokens",
            "context_compression_per_context_chars": "context_compression_per_context_chars",
            "context_compression_total_chars": "context_compression_total_chars",
            "adaptive_enabled": "adaptive_enabled",
            "adaptive_initial_top_k": "adaptive_initial_top_k",
            "adaptive_escalation_top_k": "adaptive_escalation_top_k",
            "adaptive_max_rounds": "adaptive_max_rounds",
            "adaptive_max_refinements": "adaptive_max_refinements",
            "adaptive_max_latency_ms": "adaptive_max_latency_ms",
            "adaptive_min_top_score": "adaptive_min_top_score",
            "adaptive_min_score_margin": "adaptive_min_score_margin",
            "adaptive_min_source_diversity": "adaptive_min_source_diversity",
            "adaptive_min_query_coverage": "adaptive_min_query_coverage",
            "adaptive_refinement_max_chars": "adaptive_refinement_max_chars",
            "adaptive_critique_enabled": "adaptive_critique_enabled",
            "adaptive_max_provider_calls": "adaptive_max_provider_calls",
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
            "timeout_seconds": "llm_timeout_seconds",
        },
        "dataset": {"roots": "ingest_roots", "recursive": "ingest_recursive"},
        "evaluation": {"seed": "eval_seed"},
        "observability": {
            "enabled": "otel_enabled",
            "endpoint": "otel_exporter_endpoint",
            "service_name": "otel_service_name",
            "sample_rate": "otel_sample_rate",
            "timeout_seconds": "otel_exporter_timeout_seconds",
            "capture_content": "otel_capture_content",
            "max_attribute_length": "otel_max_attribute_length",
        },
    }
    fields = set(FLAT_TO_PATH) | UNGROUPED_FIELDS
    flattened: dict[str, Any] = {}
    for key, value in document.items():
        if key in sections:
            if not isinstance(value, dict):
                raise ConfigError(f"YAML section {key} must be a mapping")  # noqa: EM102
            for nested_key, nested_value in value.items():
                if f"{key}.{nested_key}" in _RETIRED_FLAGS:
                    _warn_retired_flag(f"{key}.{nested_key}")
                    continue
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
        elif key in _RETIRED_FLAGS:
            _warn_retired_flag(key)
        elif key in fields:
            if key in _SECRET_FIELDS and value and not (isinstance(value, str) and "${" in value):
                raise ConfigError("Secrets must be supplied through the environment")
            flattened[key] = value
        else:
            raise ConfigError(f"Unknown YAML key: {key}")  # noqa: EM102

    flattened = _interpolate(flattened)
    base = path.parent
    for field in ("chroma_persist_path", "upload_dir", "audit_log_path", "embedding_cache_path"):
        if flattened.get(field):
            flattened[field] = str(_resolve_path(str(flattened[field]), base))
    if flattened.get("ingest_roots"):
        flattened["ingest_roots"] = [
            str(_resolve_path(str(root), base)) for root in flattened["ingest_roots"]
        ]
    # YAML is authored in flat field names; the model is grouped.
    return _nest(flattened)


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
    destination), enforces ``upload_max_bytes``, and treats files as temporary
    artifacts unless ``upload_retention_seconds`` is positive.

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

        **Query expansion** — When ``query_expansion_enabled`` is true, one
        additional LLM call generates bounded retrieval variants after the
        optional rewrite. The original question is always retained for variant
        retrieval; generated variants never become answer facts. Expansion is
        capped by the variant, query-length, and candidate-budget settings.

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
        env_settings: PydanticBaseSettingsSource,  # noqa: ARG003 — replaced by _FlatEnvSource
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003 — replaced by _FlatDotenvSource
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # The flat sources replace pydantic's own env/dotenv handling: the public
        # names are flat but the model is grouped, so they must be regrouped before
        # validation. Precedence is unchanged: init, env, .env, YAML, file secrets.
        return (
            init_settings,
            _FlatEnvSource(settings_cls),
            _FlatDotenvSource(settings_cls),
            _YamlSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="before")
    @classmethod
    def _accept_flat_values(cls, data: Any) -> Any:
        """Allow ``Settings(hyde_enabled=True)`` alongside the grouped form.

        Direct construction with flat keyword arguments is the documented way to
        build settings in tests and callers; regrouping here keeps that working
        without every call site learning the group layout.
        """
        if not isinstance(data, dict):
            return data
        if any(key in FLAT_TO_PATH for key in data):
            return _nest(data)
        return data

    ollama: OllamaSettings = OllamaSettings()
    chroma: ChromaSettings = ChromaSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    ingest: IngestSettings = IngestSettings()
    upload: UploadSettings = UploadSettings()
    audit: AuditSettings = AuditSettings()
    ocr: OcrSettings = OcrSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    query_cache: QueryCacheSettings = QueryCacheSettings()
    api: ApiSettings = ApiSettings()
    llm: LlmSettings = LlmSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    # Ungrouped: these belong to no bounded feature.
    log_level: str = "INFO"
    # Written to every chunk's metadata and usable as a QueryRequest.metadata_filter
    # key ({"tenant_id": "..."}). Empty = untagged (single-tenant, the common case).
    tenant_id: str = ""
    eval_seed: int = 42

    # --- Flat accessors -------------------------------------------------
    # Every documented flat name stays readable as ``settings.<name>``, so
    # grouping the model required no call-site changes. These are properties,
    # not fields, which is what keeps ``resolved_snapshot`` grouped.

    @property
    def ollama_base_url(self) -> str:
        """Flat accessor for ``ollama.base_url``."""
        return self.ollama.base_url

    @property
    def ollama_embed_model(self) -> str:
        """Flat accessor for ``ollama.embed_model``."""
        return self.ollama.embed_model

    @property
    def ollama_llm_model(self) -> str:
        """Flat accessor for ``ollama.llm_model``."""
        return self.ollama.llm_model

    @property
    def llm_temperature(self) -> float | None:
        """Flat accessor for ``llm.temperature``."""
        return self.llm.temperature

    @property
    def llm_seed(self) -> int | None:
        """Flat accessor for ``llm.seed``."""
        return self.llm.seed

    @property
    def llm_timeout_seconds(self) -> float:
        """Flat accessor for ``llm.timeout_seconds``."""
        return self.llm.timeout_seconds

    @property
    def llm_backend(self) -> str:
        """Flat accessor for ``llm.backend``."""
        return self.llm.backend

    @property
    def llm_fallback_backend(self) -> str:
        """Flat accessor for ``llm.fallback_backend``."""
        return self.llm.fallback_backend

    @property
    def llm_retry_max_attempts(self) -> int:
        """Flat accessor for ``llm.retry_max_attempts``."""
        return self.llm.retry_max_attempts

    @property
    def llm_circuit_fail_max(self) -> int:
        """Flat accessor for ``llm.circuit_fail_max``."""
        return self.llm.circuit_fail_max

    @property
    def llm_circuit_reset_timeout_seconds(self) -> float:
        """Flat accessor for ``llm.circuit_reset_timeout_seconds``."""
        return self.llm.circuit_reset_timeout_seconds

    @property
    def openai_api_key(self) -> str:
        """Flat accessor for ``llm.openai_api_key``."""
        return self.llm.openai_api_key

    @property
    def openai_model(self) -> str:
        """Flat accessor for ``llm.openai_model``."""
        return self.llm.openai_model

    @property
    def anthropic_api_key(self) -> str:
        """Flat accessor for ``llm.anthropic_api_key``."""
        return self.llm.anthropic_api_key

    @property
    def anthropic_model(self) -> str:
        """Flat accessor for ``llm.anthropic_model``."""
        return self.llm.anthropic_model

    @property
    def agent_model(self) -> str:
        """Flat accessor for ``llm.agent_model``."""
        return self.llm.agent_model

    @property
    def chroma_persist_path(self) -> str:
        """Flat accessor for ``chroma.persist_path``."""
        return self.chroma.persist_path

    @property
    def chroma_collection_name(self) -> str:
        """Flat accessor for ``chroma.collection_name``."""
        return self.chroma.collection_name

    @property
    def chunk_chars(self) -> int:
        """Flat accessor for ``chunking.chars``."""
        return self.chunking.chars

    @property
    def chunk_overlap_chars(self) -> int:
        """Flat accessor for ``chunking.overlap_chars``."""
        return self.chunking.overlap_chars

    @property
    def chunking_mode(self) -> str:
        """Flat accessor for ``chunking.mode``."""
        return self.chunking.mode

    @property
    def chunk_max_chars(self) -> int:
        """Flat accessor for ``chunking.max_chars``."""
        return self.chunking.max_chars

    @property
    def chunk_min_chars(self) -> int:
        """Flat accessor for ``chunking.min_chars``."""
        return self.chunking.min_chars

    @property
    def embedding_batch_size(self) -> int:
        """Flat accessor for ``embedding.batch_size``."""
        return self.embedding.batch_size

    @property
    def embedding_provider(self) -> str:
        """Flat accessor for ``embedding.provider``."""
        return self.embedding.provider

    @property
    def embedding_timeout_seconds(self) -> float:
        """Flat accessor for ``embedding.timeout_seconds``."""
        return self.embedding.timeout_seconds

    @property
    def sentence_transformers_model(self) -> str:
        """Flat accessor for ``embedding.sentence_transformers_model``."""
        return self.embedding.sentence_transformers_model

    @property
    def embedding_model(self) -> str:
        """Flat accessor for ``embedding.model``."""
        return self.embedding.model

    @property
    def embedding_cache_path(self) -> str:
        """Flat accessor for ``embedding.cache_path``."""
        return self.embedding.cache_path

    @property
    def embedding_cache_max_entries(self) -> int:
        """Flat accessor for ``embedding.cache_max_entries``."""
        return self.embedding.cache_max_entries

    @property
    def embedding_cache_max_bytes(self) -> int:
        """Flat accessor for ``embedding.cache_max_bytes``."""
        return self.embedding.cache_max_bytes

    @property
    def embedding_cache_preprocessing_version(self) -> str:
        """Flat accessor for ``embedding.cache_preprocessing_version``."""
        return self.embedding.cache_preprocessing_version

    @property
    def embedding_cache_task_prefix(self) -> str:
        """Flat accessor for ``embedding.cache_task_prefix``."""
        return self.embedding.cache_task_prefix

    @property
    def ingest_recursive(self) -> bool:
        """Flat accessor for ``ingest.recursive``."""
        return self.ingest.recursive

    @property
    def ingest_roots(self) -> list[str]:
        """Flat accessor for ``ingest.roots``."""
        return self.ingest.roots

    @property
    def max_pending_ingest_jobs(self) -> int:
        """Flat accessor for ``ingest.max_pending_jobs``."""
        return self.ingest.max_pending_jobs

    @property
    def upload_dir(self) -> str:
        """Flat accessor for ``upload.dir``."""
        return self.upload.dir

    @property
    def upload_max_bytes(self) -> int:
        """Flat accessor for ``upload.max_bytes``."""
        return self.upload.max_bytes

    @property
    def upload_retention_seconds(self) -> float:
        """Flat accessor for ``upload.retention_seconds``."""
        return self.upload.retention_seconds

    @property
    def upload_quota_bytes(self) -> int:
        """Flat accessor for ``upload.quota_bytes``."""
        return self.upload.quota_bytes

    @property
    def audit_log_path(self) -> str:
        """Flat accessor for ``audit.log_path``."""
        return self.audit.log_path

    @property
    def audit_log_max_bytes(self) -> int:
        """Flat accessor for ``audit.log_max_bytes``."""
        return self.audit.log_max_bytes

    @property
    def audit_log_retention_seconds(self) -> float:
        """Flat accessor for ``audit.log_retention_seconds``."""
        return self.audit.log_retention_seconds

    @property
    def audit_log_metadata_only(self) -> bool:
        """Flat accessor for ``audit.log_metadata_only``."""
        return self.audit.log_metadata_only

    @property
    def audit_log_redact_content(self) -> bool:
        """Flat accessor for ``audit.log_redact_content``."""
        return self.audit.log_redact_content

    @property
    def ocr_enabled(self) -> bool:
        """Flat accessor for ``ocr.enabled``."""
        return self.ocr.enabled

    @property
    def ocr_language(self) -> str:
        """Flat accessor for ``ocr.language``."""
        return self.ocr.language

    @property
    def ocr_min_chars_per_page(self) -> int:
        """Flat accessor for ``ocr.min_chars_per_page``."""
        return self.ocr.min_chars_per_page

    @property
    def rag_top_k(self) -> int:
        """Flat accessor for ``retrieval.top_k``."""
        return self.retrieval.top_k

    @property
    def retriever_plugin(self) -> str:
        """Flat accessor for ``retrieval.plugin``."""
        return self.retrieval.plugin

    @property
    def rag_min_context_score(self) -> float:
        """Flat accessor for ``retrieval.min_context_score``."""
        return self.retrieval.min_context_score

    @property
    def retrieval_mode(self) -> str:
        """Flat accessor for ``retrieval.mode``."""
        return self.retrieval.mode

    @property
    def bm25_weight(self) -> float:
        """Flat accessor for ``retrieval.bm25_weight``."""
        return self.retrieval.bm25_weight

    @property
    def rrf_k(self) -> int:
        """Flat accessor for ``retrieval.rrf_k``."""
        return self.retrieval.rrf_k

    @property
    def freshness_half_life_days(self) -> float:
        """Flat accessor for ``retrieval.freshness_half_life_days``."""
        return self.retrieval.freshness_half_life_days

    @property
    def freshness_weight(self) -> float:
        """Flat accessor for ``retrieval.freshness_weight``."""
        return self.retrieval.freshness_weight

    @property
    def parent_expansion_enabled(self) -> bool:
        """Flat accessor for ``retrieval.parent_expansion_enabled``."""
        return self.retrieval.parent_expansion_enabled

    @property
    def query_rewrite_enabled(self) -> bool:
        """Flat accessor for ``retrieval.query_rewrite_enabled``."""
        return self.retrieval.query_rewrite_enabled

    @property
    def retrieval_experiment_mode(self) -> str:
        """Flat accessor for ``retrieval.experiment_mode``."""
        return self.retrieval.experiment_mode

    @property
    def rag_system_prompt(self) -> str:
        """Flat accessor for ``retrieval.system_prompt``."""
        return self.retrieval.system_prompt

    @property
    def query_expansion_enabled(self) -> bool:
        """Flat accessor for ``retrieval.expansion.enabled``."""
        return self.retrieval.expansion.enabled

    @property
    def query_expansion_max_variants(self) -> int:
        """Flat accessor for ``retrieval.expansion.max_variants``."""
        return self.retrieval.expansion.max_variants

    @property
    def query_expansion_max_query_chars(self) -> int:
        """Flat accessor for ``retrieval.expansion.max_query_chars``."""
        return self.retrieval.expansion.max_query_chars

    @property
    def query_expansion_candidate_budget(self) -> int:
        """Flat accessor for ``retrieval.expansion.candidate_budget``."""
        return self.retrieval.expansion.candidate_budget

    @property
    def hyde_enabled(self) -> bool:
        """Flat accessor for ``retrieval.hyde.enabled``."""
        return self.retrieval.hyde.enabled

    @property
    def hyde_model(self) -> str:
        """Flat accessor for ``retrieval.hyde.model``."""
        return self.retrieval.hyde.model

    @property
    def hyde_timeout_seconds(self) -> float:
        """Flat accessor for ``retrieval.hyde.timeout_seconds``."""
        return self.retrieval.hyde.timeout_seconds

    @property
    def hyde_input_max_chars(self) -> int:
        """Flat accessor for ``retrieval.hyde.input_max_chars``."""
        return self.retrieval.hyde.input_max_chars

    @property
    def hyde_output_max_chars(self) -> int:
        """Flat accessor for ``retrieval.hyde.output_max_chars``."""
        return self.retrieval.hyde.output_max_chars

    @property
    def hyde_output_max_tokens(self) -> int:
        """Flat accessor for ``retrieval.hyde.output_max_tokens``."""
        return self.retrieval.hyde.output_max_tokens

    @property
    def hyde_lexical_input(self) -> str:
        """Flat accessor for ``retrieval.hyde.lexical_input``."""
        return self.retrieval.hyde.lexical_input

    @property
    def hyde_log_content(self) -> bool:
        """Flat accessor for ``retrieval.hyde.log_content``."""
        return self.retrieval.hyde.log_content

    @property
    def rerank_enabled(self) -> bool:
        """Flat accessor for ``retrieval.rerank.enabled``."""
        return self.retrieval.rerank.enabled

    @property
    def rerank_model(self) -> str:
        """Flat accessor for ``retrieval.rerank.model``."""
        return self.retrieval.rerank.model

    @property
    def rerank_fetch_k(self) -> int:
        """Flat accessor for ``retrieval.rerank.fetch_k``."""
        return self.retrieval.rerank.fetch_k

    @property
    def context_compression_candidate_count(self) -> int:
        """Flat accessor for ``retrieval.compression.candidate_count``."""
        return self.retrieval.compression.candidate_count

    @property
    def context_compression_max_contexts(self) -> int:
        """Flat accessor for ``retrieval.compression.max_contexts``."""
        return self.retrieval.compression.max_contexts

    @property
    def context_compression_per_context_tokens(self) -> int:
        """Flat accessor for ``retrieval.compression.per_context_tokens``."""
        return self.retrieval.compression.per_context_tokens

    @property
    def context_compression_total_tokens(self) -> int:
        """Flat accessor for ``retrieval.compression.total_tokens``."""
        return self.retrieval.compression.total_tokens

    @property
    def context_compression_per_context_chars(self) -> int:
        """Flat accessor for ``retrieval.compression.per_context_chars``."""
        return self.retrieval.compression.per_context_chars

    @property
    def context_compression_total_chars(self) -> int:
        """Flat accessor for ``retrieval.compression.total_chars``."""
        return self.retrieval.compression.total_chars

    @property
    def context_compression_reserved_prompt_tokens(self) -> int:
        """Flat accessor for ``retrieval.compression.reserved_prompt_tokens``."""
        return self.retrieval.compression.reserved_prompt_tokens

    @property
    def context_compression_reserved_answer_tokens(self) -> int:
        """Flat accessor for ``retrieval.compression.reserved_answer_tokens``."""
        return self.retrieval.compression.reserved_answer_tokens

    @property
    def llm_context_window_tokens(self) -> int:
        """Flat accessor for ``retrieval.compression.llm_context_window_tokens``."""
        return self.retrieval.compression.llm_context_window_tokens

    @property
    def adaptive_enabled(self) -> bool:
        """Flat accessor for ``retrieval.adaptive.enabled``."""
        return self.retrieval.adaptive.enabled

    @property
    def adaptive_initial_top_k(self) -> int:
        """Flat accessor for ``retrieval.adaptive.initial_top_k``."""
        return self.retrieval.adaptive.initial_top_k

    @property
    def adaptive_escalation_top_k(self) -> int:
        """Flat accessor for ``retrieval.adaptive.escalation_top_k``."""
        return self.retrieval.adaptive.escalation_top_k

    @property
    def adaptive_max_rounds(self) -> int:
        """Flat accessor for ``retrieval.adaptive.max_rounds``."""
        return self.retrieval.adaptive.max_rounds

    @property
    def adaptive_max_refinements(self) -> int:
        """Flat accessor for ``retrieval.adaptive.max_refinements``."""
        return self.retrieval.adaptive.max_refinements

    @property
    def adaptive_max_latency_ms(self) -> float:
        """Flat accessor for ``retrieval.adaptive.max_latency_ms``."""
        return self.retrieval.adaptive.max_latency_ms

    @property
    def adaptive_min_top_score(self) -> float:
        """Flat accessor for ``retrieval.adaptive.min_top_score``."""
        return self.retrieval.adaptive.min_top_score

    @property
    def adaptive_min_score_margin(self) -> float:
        """Flat accessor for ``retrieval.adaptive.min_score_margin``."""
        return self.retrieval.adaptive.min_score_margin

    @property
    def adaptive_min_source_diversity(self) -> int:
        """Flat accessor for ``retrieval.adaptive.min_source_diversity``."""
        return self.retrieval.adaptive.min_source_diversity

    @property
    def adaptive_min_query_coverage(self) -> float:
        """Flat accessor for ``retrieval.adaptive.min_query_coverage``."""
        return self.retrieval.adaptive.min_query_coverage

    @property
    def adaptive_refinement_max_chars(self) -> int:
        """Flat accessor for ``retrieval.adaptive.refinement_max_chars``."""
        return self.retrieval.adaptive.refinement_max_chars

    @property
    def adaptive_critique_enabled(self) -> bool:
        """Flat accessor for ``retrieval.adaptive.critique_enabled``."""
        return self.retrieval.adaptive.critique_enabled

    @property
    def adaptive_max_provider_calls(self) -> int:
        """Flat accessor for ``retrieval.adaptive.max_provider_calls``."""
        return self.retrieval.adaptive.max_provider_calls

    @property
    def query_cache_ttl_seconds(self) -> float:
        """Flat accessor for ``query_cache.ttl_seconds``."""
        return self.query_cache.ttl_seconds

    @property
    def query_cache_maxsize(self) -> int:
        """Flat accessor for ``query_cache.maxsize``."""
        return self.query_cache.maxsize

    @property
    def api_host(self) -> str:
        """Flat accessor for ``api.host``."""
        return self.api.host

    @property
    def api_port(self) -> int:
        """Flat accessor for ``api.port``."""
        return self.api.port

    @property
    def api_key(self) -> str:
        """Flat accessor for ``api.key``."""
        return self.api.key

    @property
    def otel_enabled(self) -> bool:
        """Flat accessor for ``observability.otel_enabled``."""
        return self.observability.otel_enabled

    @property
    def otel_exporter_endpoint(self) -> str:
        """Flat accessor for ``observability.exporter_endpoint``."""
        return self.observability.exporter_endpoint

    @property
    def otel_service_name(self) -> str:
        """Flat accessor for ``observability.service_name``."""
        return self.observability.service_name

    @property
    def otel_sample_rate(self) -> float:
        """Flat accessor for ``observability.sample_rate``."""
        return self.observability.sample_rate

    @property
    def otel_exporter_timeout_seconds(self) -> float:
        """Flat accessor for ``observability.exporter_timeout_seconds``."""
        return self.observability.exporter_timeout_seconds

    @property
    def otel_exporter_retry_count(self) -> int:
        """Flat accessor for ``observability.exporter_retry_count``."""
        return self.observability.exporter_retry_count

    @property
    def otel_capture_content(self) -> bool:
        """Flat accessor for ``observability.capture_content``."""
        return self.observability.capture_content

    @property
    def otel_max_attribute_length(self) -> int:
        """Flat accessor for ``observability.max_attribute_length``."""
        return self.observability.max_attribute_length

    @property
    def langfuse_enabled(self) -> bool:
        """Flat accessor for ``observability.langfuse_enabled``."""
        return self.observability.langfuse_enabled

    @property
    def effective_embedding_model(self) -> str:
        """Resolve the new model name while preserving legacy Ollama deployments."""
        return self.embedding.model.strip() or self.ollama.embed_model

    def with_overrides(self, **flat_values: Any) -> Settings:
        """Return a copy with flat field names applied to their grouped homes.

        ``model_copy(update={"hyde_enabled": True})`` would silently do nothing,
        because the flat names are properties rather than fields. Use this instead
        wherever a derived settings object is needed.
        """
        merged = self.model_dump()
        for name, value in flat_values.items():
            path = FLAT_TO_PATH.get(name)
            if path is None:
                merged[name] = value
                continue
            *sections, leaf = path.split(".")
            cursor: Any = merged
            for section in sections:
                cursor = cursor[section]
            cursor[leaf] = value
        return type(self).model_validate(merged)

    def resolved_snapshot(self) -> dict[str, Any]:
        """Return deterministic, non-secret configuration provenance.

        The shape mirrors the grouped model (ADR 037). Consumers that want the
        documented flat names should use :meth:`flat_snapshot`.
        """
        snapshot = self.model_dump(mode="json")
        for field in _SECRET_FIELDS:
            _redact_path(snapshot, FLAT_TO_PATH.get(field, field), "<redacted>")
        for field in _PATH_FIELDS:
            _redact_path(snapshot, FLAT_TO_PATH.get(field, field), "<path>")
        return snapshot

    def flat_snapshot(self) -> dict[str, Any]:
        """Return the same provenance keyed by the documented flat field names."""
        nested = self.resolved_snapshot()
        flat: dict[str, Any] = {}
        for name, path in FLAT_TO_PATH.items():
            cursor: Any = nested
            for part in path.split("."):
                if not isinstance(cursor, dict) or part not in cursor:
                    cursor = None
                    break
                cursor = cursor[part]
            flat[name] = cursor
        for name in UNGROUPED_FIELDS:
            flat[name] = nested.get(name)
        return flat


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
    supplied = dict(cli_overrides or {})
    for retired in sorted(set(supplied) & _RETIRED_FLAGS):
        _warn_retired_flag(retired)
        supplied.pop(retired)
    overrides = tuple(sorted(supplied.items()))
    unknown = set(supplied) - set(FLAT_TO_PATH) - UNGROUPED_FIELDS - set(Settings.model_fields)
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

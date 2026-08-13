"""Grouped configuration sub-models backing the flat :class:`~localrag.settings.Settings`.

Each model owns one bounded feature's fields **and its validation**, replacing the
single ~105-line ``validate_configuration`` that previously carried
``# noqa: C901, PLR0912, PLR0915``. See
`docs/adr/037-grouped-configuration-model.md`.

The flat, documented environment variable names remain the public contract:
``Settings`` exposes a read-only property per field and a regrouping settings
source maps ``HYDE_ENABLED`` onto ``hyde.enabled``. Nothing here is addressed by
environment variables directly.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from localrag.rag.prompt import DEFAULT_SYSTEM_PROMPT

# Defaults for Ollama model tags (`ollama pull` / `ollama list`).
# Keep in sync with docs and API examples.
DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_LLM_MODEL = "gemma3:4b"

_SUPPORTED_BACKENDS = frozenset({"ollama", "openai", "anthropic"})


class OllamaSettings(BaseModel):
    """Ollama HTTP endpoint and model tags as shown by ``ollama list``."""

    base_url: str = "http://localhost:11434"
    embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL
    llm_model: str = DEFAULT_OLLAMA_LLM_MODEL


class ChromaSettings(BaseModel):
    """On-disk vector store location and collection namespace."""

    persist_path: str = "./data/chroma"
    collection_name: str = "localrag"


class ChunkingSettings(BaseModel):
    """Chunk sizing shared by every parser."""

    chars: int = 512
    overlap_chars: int = 150
    mode: str = "structural"
    max_chars: int = 1200
    min_chars: int = 200

    @model_validator(mode="after")
    def _validate(self) -> ChunkingSettings:
        if self.min_chars > self.max_chars:
            raise ValueError("chunk_min_chars must be less than or equal to chunk_max_chars")
        if self.mode not in {"fixed", "structural", "recursive"}:
            raise ValueError("chunking_mode must be 'fixed', 'structural', or 'recursive'")
        return self


class EmbeddingSettings(BaseModel):
    """Embedding provider selection and the always-on ingestion cache (ADR 036)."""

    batch_size: int = 32
    provider: str = "ollama"
    timeout_seconds: float = 120.0
    sentence_transformers_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    model: str = ""
    cache_path: str = "./data/embedding-cache"
    cache_max_entries: int = 10_000
    cache_max_bytes: int = 1_000_000_000
    cache_preprocessing_version: str = "1"
    cache_task_prefix: str = ""

    @model_validator(mode="after")
    def _validate(self) -> EmbeddingSettings:
        if self.cache_max_entries < 1 or self.cache_max_bytes < 1:
            raise ValueError("embedding cache limits must be positive")
        return self


class IngestSettings(BaseModel):
    """Directory ingest defaults and the HTTP ingest path allow-list."""

    recursive: bool = True
    roots: list[str] = []
    # Caps concurrently pending/running async ingest jobs; further submissions get 429.
    max_pending_jobs: int = 10


class UploadSettings(BaseModel):
    """Upload limits. Uploads are temporary ingest artifacts unless retained."""

    dir: str = "./data/uploads"
    max_bytes: int = 100_000_000
    retention_seconds: float = 0.0
    quota_bytes: int = 1_000_000_000

    @model_validator(mode="after")
    def _validate(self) -> UploadSettings:
        if self.max_bytes < 1 or self.quota_bytes < 1:
            raise ValueError("upload limits must be positive")
        if self.retention_seconds < 0:
            raise ValueError("upload_retention_seconds must not be negative")
        return self


class AuditSettings(BaseModel):
    """Query-audit log location, rotation, and redaction."""

    log_path: str = ""
    log_max_bytes: int = 10_000_000
    log_retention_seconds: float = 2_592_000.0
    log_metadata_only: bool = False
    log_redact_content: bool = False

    @model_validator(mode="after")
    def _validate(self) -> AuditSettings:
        if self.log_max_bytes < 1 or self.log_retention_seconds < 0:
            raise ValueError("audit log limits are invalid")
        return self


class OcrSettings(BaseModel):
    """Tesseract OCR fallback for PDF pages with unreliable text extraction."""

    enabled: bool = True
    language: str = "eng"
    min_chars_per_page: int = 20


class QueryExpansionSettings(BaseModel):
    """Bounded retrieval-variant expansion. Hard caps apply independently."""

    enabled: bool = False
    max_variants: int = 4
    max_query_chars: int = 500
    candidate_budget: int = 40

    @model_validator(mode="after")
    def _validate(self) -> QueryExpansionSettings:
        if self.max_variants < 1:
            raise ValueError("query_expansion_max_variants must be at least 1")
        if self.max_query_chars < 1:
            raise ValueError("query_expansion_max_query_chars must be at least 1")
        if self.candidate_budget < 1:
            raise ValueError("query_expansion_candidate_budget must be at least 1")
        return self


class HydeSettings(BaseModel):
    """Hypothetical-document retrieval, an explicit experiment arm (ADR 025)."""

    enabled: bool = False
    model: str = ""
    timeout_seconds: float = 30.0
    input_max_chars: int = 2000
    output_max_chars: int = 4000
    output_max_tokens: int = 512
    lexical_input: str = "original"
    log_content: bool = False

    @model_validator(mode="after")
    def _validate(self) -> HydeSettings:
        if self.lexical_input not in {"original", "rewritten"}:
            raise ValueError("hyde_lexical_input must be 'original' or 'rewritten'")
        limits = (
            self.timeout_seconds,
            self.input_max_chars,
            self.output_max_chars,
            self.output_max_tokens,
        )
        if min(limits) <= 0:
            raise ValueError("HyDE limits must be positive")
        return self


class RerankSettings(BaseModel):
    """Optional cross-encoder reranking (requires ``uv sync --extra rerank``)."""

    enabled: bool = False
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fetch_k: int = 20


class ContextCompressionSettings(BaseModel):
    """Always-applied extractive compression budgets (ADR 022, amended by ADR 036).

    The token counter is the documented whitespace-token approximation; the total
    context budget must fit inside the reserved generation window.
    """

    candidate_count: int = 20
    max_contexts: int = 5
    per_context_tokens: int = 256
    total_tokens: int = 1024
    per_context_chars: int = 4000
    total_chars: int = 16000
    reserved_prompt_tokens: int = 512
    reserved_answer_tokens: int = 512
    llm_context_window_tokens: int = 4096

    @model_validator(mode="after")
    def _validate(self) -> ContextCompressionSettings:
        values = (
            self.candidate_count,
            self.max_contexts,
            self.per_context_tokens,
            self.total_tokens,
            self.per_context_chars,
            self.total_chars,
            self.reserved_prompt_tokens,
            self.reserved_answer_tokens,
            self.llm_context_window_tokens,
        )
        if any(value < 1 for value in values):
            raise ValueError("context compression budgets must be positive")
        available = (
            self.llm_context_window_tokens
            - self.reserved_prompt_tokens
            - self.reserved_answer_tokens
        )
        if self.total_tokens > available:
            raise ValueError("context compression total token budget exceeds model context window")
        return self


class AdaptiveSettings(BaseModel):
    """Bounded adaptive retrieval (ADR 023).

    Thresholds are corpus-tuned evidence heuristics, not calibrated model
    confidence; the hard caps prevent agent-like unbounded loops.
    """

    enabled: bool = False
    initial_top_k: int = 3
    escalation_top_k: int = 8
    max_rounds: int = 3
    max_refinements: int = 1
    max_latency_ms: float = 10_000.0
    min_top_score: float = 0.35
    min_score_margin: float = 0.02
    min_source_diversity: int = 1
    min_query_coverage: float = 0.2
    refinement_max_chars: int = 500
    critique_enabled: bool = False
    max_provider_calls: int = 2

    @model_validator(mode="after")
    def _validate(self) -> AdaptiveSettings:
        if self.initial_top_k < 1 or self.escalation_top_k < self.initial_top_k:
            raise ValueError("adaptive retrieval k settings are invalid")
        if self.max_rounds < 1 or self.max_refinements < 0:
            raise ValueError("adaptive retrieval budgets are invalid")
        if self.max_latency_ms <= 0 or self.refinement_max_chars < 1:
            raise ValueError("adaptive retrieval limits must be positive")
        if self.max_provider_calls < 0:
            raise ValueError("adaptive provider call budget must not be negative")
        return self


class ClaimFilterSettings(BaseModel):
    """Bounded scope-applicability filtering of retrieved contexts (ADR 041).

    Off by default: it costs one extra provider call per query, and the prompt-level
    scope instruction already covers the common case without one.
    """

    enabled: bool = False
    model: str = ""
    timeout_seconds: float = 30.0
    input_max_chars: int = 1000
    output_max_tokens: int = 256
    log_content: bool = False

    @model_validator(mode="after")
    def _validate(self) -> ClaimFilterSettings:
        limits = (self.timeout_seconds, self.input_max_chars, self.output_max_tokens)
        if min(limits) <= 0:
            raise ValueError("claim filter limits must be positive")
        return self


class RetrievalSettings(BaseModel):
    """Ranking, fusion, recency, and the retrieval feature sub-models."""

    top_k: int = 5
    plugin: str = "builtin"
    min_context_score: float = 0.0
    mode: str = "hybrid"
    bm25_weight: float = 0.5
    rrf_k: int = 60
    freshness_half_life_days: float = 30.0
    freshness_weight: float = 0.15
    parent_expansion_enabled: bool = True
    query_rewrite_enabled: bool = False
    experiment_mode: str = "auto"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    expansion: QueryExpansionSettings = QueryExpansionSettings()
    hyde: HydeSettings = HydeSettings()
    rerank: RerankSettings = RerankSettings()
    compression: ContextCompressionSettings = ContextCompressionSettings()
    adaptive: AdaptiveSettings = AdaptiveSettings()
    claim_filter: ClaimFilterSettings = ClaimFilterSettings()

    @model_validator(mode="after")
    def _validate(self) -> RetrievalSettings:
        if self.mode not in {"hybrid", "vector"}:
            raise ValueError("retrieval_mode must be 'hybrid' or 'vector'")
        if not 0 <= self.bm25_weight <= 1:
            raise ValueError("bm25_weight must be between 0 and 1")
        if self.experiment_mode not in {"auto", "baseline", "rewrite", "hyde", "rewrite+hyde"}:
            raise ValueError("retrieval_experiment_mode is invalid")
        return self


class QueryCacheSettings(BaseModel):
    """In-process TTL cache for repeated queries (``0`` disables; no external cache)."""

    ttl_seconds: float = 0.0
    maxsize: int = 256


class ApiSettings(BaseModel):
    """Uvicorn bind address and optional ``X-API-Key`` enforcement."""

    host: str = "0.0.0.0"  # nosec B104 — configurable bind address, default intentional
    port: int = 8000
    # Empty (default) disables authentication.
    key: str = ""


class LlmSettings(BaseModel):
    """Provider selection, sampling, and resilience budgets.

    ``temperature`` and ``seed`` default to unset so Ollama keeps its per-model
    defaults; set them to make answers reproducible.
    """

    backend: str = "ollama"
    temperature: float | None = None
    seed: int | None = None
    timeout_seconds: float = 180.0
    fallback_backend: str = ""
    retry_max_attempts: int = 3
    circuit_fail_max: int = 5
    circuit_reset_timeout_seconds: float = 30.0

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    agent_model: str = "claude-haiku-4-5"

    @model_validator(mode="after")
    def _validate(self) -> LlmSettings:
        if self.backend not in _SUPPORTED_BACKENDS:
            message = (
                f"llm_backend must be one of {sorted(_SUPPORTED_BACKENDS)}, got {self.backend!r}"
            )
            raise ValueError(message)
        if self.fallback_backend and self.fallback_backend not in _SUPPORTED_BACKENDS:
            message = (
                "llm_fallback_backend must be empty or one of "
                f"{sorted(_SUPPORTED_BACKENDS)}, got {self.fallback_backend!r}"
            )
            raise ValueError(message)
        if self.fallback_backend == self.backend:
            raise ValueError("llm_fallback_backend must differ from llm_backend")
        return self


class ObservabilitySettings(BaseModel):
    """Optional OpenTelemetry export (ADR 030). Content is never exported by default."""

    otel_enabled: bool = False
    exporter_endpoint: str = "http://localhost:4318"
    service_name: str = "localrag"
    sample_rate: float = 1.0
    exporter_timeout_seconds: float = 10.0
    exporter_retry_count: int = 3
    capture_content: bool = False
    max_attribute_length: int = 256
    langfuse_enabled: bool = False

    @model_validator(mode="after")
    def _validate(self) -> ObservabilitySettings:
        if not 0 <= self.sample_rate <= 1:
            raise ValueError("otel_sample_rate must be between 0 and 1")
        if self.max_attribute_length < 1:
            raise ValueError("otel_max_attribute_length must be at least 1")
        return self

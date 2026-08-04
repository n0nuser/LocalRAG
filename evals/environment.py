"""Environment capture for reproducible eval runs.

Every result file records enough about the machine, code, models, and config
to answer "why does this number differ from last week's?" without guesswork.

This is an **input/config reproducibility** contract, not a model-output
determinism guarantee: two runs with identical dataset selection, seed, and
settings are provable identical in *what they asked the model*, never in
*what the model said back*. See docs/reproducibility.md for the levels this
distinguishes and the field reference.

Nothing here raises: a benchmark must not fail because `git` is missing,
Ollama is down, or there's no GPU. Every probe that can be legitimately
absent uses `Capability` — value + status + reason — instead of a bare
`None`, so "field wasn't checked", "checked and unsupported", and "checked,
should exist, and doesn't" are never conflated.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from localrag.settings import Settings

REPO_ROOT = Path(__file__).parent.parent
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

# Bump when a field is removed or its meaning changes incompatibly. Purely
# additive fields do not require a bump — a reader keyed on this version can
# still trust every field that existed at that version.
METADATA_SCHEMA_VERSION = 2

# Packages whose version can materially change judge/embedding/generation
# behavior. uv_lock_sha256 already pins the full dependency graph; this list
# exists so a human can see *what* changed without diffing the lockfile.
TRACKED_PACKAGES = ("ragas", "openai", "httpx", "pydantic", "chromadb")

# Settings that change eval numbers. A full Settings dump would bury these in
# host-specific noise (paths, ports) and leak secrets, so the snapshot is an
# explicit allowlist.
SNAPSHOT_SETTINGS_FIELDS = (
    "ollama_embed_model",
    "ollama_llm_model",
    "llm_temperature",
    "llm_seed",
    "chunk_chars",
    "chunk_overlap_chars",
    "chunking_mode",
    "chunk_max_chars",
    "chunk_min_chars",
    "rag_top_k",
    "rag_min_context_score",
    "retrieval_mode",
    "bm25_weight",
    "rrf_k",
    "freshness_half_life_days",
    "freshness_weight",
    "parent_expansion_enabled",
    "query_rewrite_enabled",
    "retrieval_experiment_mode",
    "hyde_enabled",
    "hyde_model",
    "hyde_timeout_seconds",
    "hyde_input_max_chars",
    "hyde_output_max_chars",
    "hyde_output_max_tokens",
    "hyde_lexical_input",
    "rerank_enabled",
    "rerank_model",
    "rerank_fetch_k",
)

SeedSource = Literal["cli", "config", "default"]
CapabilityStatus = Literal["available", "unsupported", "unavailable"]

# Random operations in the eval path and whether the run's seed controls them.
# Kept as data (not comments) so it round-trips into the result file — a
# reader should never have to go read source to know what was/wasn't seeded.
SEED_COVERAGE: dict[str, bool] = {
    "record_downsampling": True,  # evals/run_evals.py::_select_records
    "judge_llm_sampling": True,  # temperature=0.0 + seed passed to llm_factory
    "answering_model_sampling": False,  # controlled by LLM_TEMPERATURE/LLM_SEED, not --seed
    "embedding_computation": False,  # deterministic given fixed input; no seed concept applies
}


@dataclass
class Capability:
    """A metadata value that may not exist, with an explicit reason why.

    ``status`` distinguishes three cases a bare ``None`` conflates:
    - ``available``: ``value`` is populated.
    - ``unsupported``: the provider/platform has no concept of this field
      (e.g. no GPU present) — not a failure, just not applicable here.
    - ``unavailable``: the field should exist but couldn't be read (e.g.
      Ollama unreachable, git not installed) — a real gap, worth noticing.
    """

    value: Any
    status: CapabilityStatus
    reason: str | None = None

    @classmethod
    def ok(cls, value: Any) -> Capability:
        return cls(value=value, status="available", reason=None)

    @classmethod
    def unsupported(cls, reason: str) -> Capability:
        return cls(value=None, status="unsupported", reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> Capability:
        return cls(value=None, status="unavailable", reason=reason)


@dataclass
class RunMetadata:
    """Provenance for a single eval run, embedded in the results JSON."""

    metadata_schema_version: int
    seed: int
    seed_source: SeedSource
    seed_coverage: dict[str, bool]

    git_sha: Capability
    git_dirty: Capability
    uv_lock_sha256: Capability
    package_versions: dict[str, str]

    python_version: str
    platform_summary: str
    cpu_count: Capability
    total_ram_gb: Capability
    gpu: Capability

    judge_model: str
    judge_model_digest: Capability
    embedding_model: str
    embedding_model_digest: Capability

    settings_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_seed(cli_seed: int | None, env_seed_var: str = "EVAL_SEED") -> tuple[int, SeedSource]:
    """Resolve the effective seed and where it came from.

    Precedence: CLI flag > ``EVAL_SEED`` env var > built-in default (42).
    An unparseable ``EVAL_SEED`` is a configuration error, not something to
    silently ignore.
    """
    if cli_seed is not None:
        return cli_seed, "cli"

    env_value = os.environ.get(env_seed_var)
    if env_value is not None:
        try:
            return int(env_value), "config"
        except ValueError as exc:
            message = f"{env_seed_var}={env_value!r} is not a valid integer seed"
            raise ValueError(message) from exc

    return 42, "default"


def _run_git(*args: str) -> str | None:
    """Return stdout of a git command, or ``None`` if git/the repo is unavailable."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607 — git resolved from PATH, fixed argv
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_sha() -> Capability:
    sha = _run_git("rev-parse", "HEAD")
    if sha is None:
        return Capability.unavailable("git not installed, or this is not a git checkout")
    return Capability.ok(sha)


def _git_dirty() -> Capability:
    """Whether the working tree has uncommitted changes (results not tied to a clean commit)."""
    status = _run_git("status", "--porcelain")
    if status is None:
        return Capability.unavailable("git not installed, or this is not a git checkout")
    return Capability.ok(bool(status))


def _uv_lock_sha256() -> Capability:
    """Hash the lockfile so dependency drift is visible across runs."""
    try:
        return Capability.ok(hashlib.sha256(UV_LOCK_PATH.read_bytes()).hexdigest())
    except OSError:
        return Capability.unavailable(f"could not read {UV_LOCK_PATH}")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _cpu_count() -> Capability:
    count = os.cpu_count()
    if count is None:
        return Capability.unavailable("os.cpu_count() returned None")
    return Capability.ok(count)


def _total_ram_gb() -> Capability:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return Capability.unsupported("os.sysconf(SC_PHYS_PAGES) not available on this platform")
    return Capability.ok(round(page_size * page_count / 1024**3, 1))


def _gpu_info() -> Capability:
    """GPU name/driver/memory via ``nvidia-smi``, when present.

    Absence of an NVIDIA GPU is the common case (most eval runs are CPU or
    Ollama-managed), so that path is ``unsupported`` rather than
    ``unavailable`` — it isn't a gap, it's the expected shape of most hosts.
    """
    try:
        result = subprocess.run(
            [  # noqa: S607 — nvidia-smi resolved from PATH, fixed argv
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return Capability.unsupported(
            "nvidia-smi not found — no NVIDIA GPU, or drivers not installed"
        )
    if result.returncode != 0 or not result.stdout.strip():
        return Capability.unsupported("nvidia-smi present but reported no GPU")

    gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    return Capability.ok(gpus)


def _ollama_model_digests(ollama_url: str) -> dict[str, str] | None:
    """Map model name -> digest from ``GET /api/tags``, or ``None`` if Ollama is unreachable."""
    try:
        resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return None

    digests: dict[str, str] = {}
    for model in body.get("models", []):
        name = model.get("name")
        digest = model.get("digest")
        if isinstance(name, str) and isinstance(digest, str):
            digests[name] = digest
    return digests


def _model_digest(digests: dict[str, str] | None, model: str, *, ollama_url: str) -> Capability:
    """Look up a model digest, tolerating the implicit ``:latest`` tag.

    Digests are required-when-available per #83: Ollama can always expose a
    digest for a pulled model, so any miss here is ``unavailable`` (Ollama
    down, or the model isn't pulled) rather than ``unsupported`` — this
    provider never lacks the *capability*, at most the current data.
    """
    if digests is None:
        return Capability.unavailable(f"could not reach Ollama at {ollama_url}")
    if model in digests:
        return Capability.ok(digests[model])
    if ":" not in model and f"{model}:latest" in digests:
        return Capability.ok(digests[f"{model}:latest"])
    return Capability.unavailable(f"model {model!r} not found in `ollama list` output")


def _settings_snapshot() -> dict[str, Any]:
    """Allowlisted settings values, or ``{}`` if settings can't be loaded."""
    try:
        settings = Settings()
    except Exception:  # snapshot is best-effort metadata; never fail the run
        return {}
    return {
        field_name: getattr(settings, field_name, None)
        for field_name in SNAPSHOT_SETTINGS_FIELDS
        if hasattr(settings, field_name)
    }


def capture_run_metadata(
    *,
    seed: int,
    seed_source: SeedSource,
    judge_model: str,
    embedding_model: str,
    ollama_url: str,
) -> RunMetadata:
    """Collect provenance for the current eval run. Never raises."""
    digests = _ollama_model_digests(ollama_url)
    return RunMetadata(
        metadata_schema_version=METADATA_SCHEMA_VERSION,
        seed=seed,
        seed_source=seed_source,
        seed_coverage=dict(SEED_COVERAGE),
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        uv_lock_sha256=_uv_lock_sha256(),
        package_versions=_package_versions(),
        python_version=sys.version.split()[0],
        platform_summary=platform.platform(),
        cpu_count=_cpu_count(),
        total_ram_gb=_total_ram_gb(),
        gpu=_gpu_info(),
        judge_model=judge_model,
        judge_model_digest=_model_digest(digests, judge_model, ollama_url=ollama_url),
        embedding_model=embedding_model,
        embedding_model_digest=_model_digest(digests, embedding_model, ollama_url=ollama_url),
        settings_snapshot=_settings_snapshot(),
    )

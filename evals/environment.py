"""Environment capture for reproducible eval runs.

Every result file records enough about the machine, code, and models to answer
"why does this number differ from last week's?" without guesswork. Nothing here
raises: a benchmark must not fail because `git` is missing or Ollama is down, so
every probe degrades to ``None`` and the run continues.

See `docs/reproducibility.md` for what is and isn't reproducible.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from localrag.settings import Settings

REPO_ROOT = Path(__file__).parent.parent
UV_LOCK_PATH = REPO_ROOT / "uv.lock"

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
    "rerank_enabled",
    "rerank_model",
    "rerank_fetch_k",
)


@dataclass
class RunMetadata:
    """Provenance for a single eval run, embedded in the results JSON."""

    seed: int
    git_sha: str | None
    git_dirty: bool | None
    uv_lock_sha256: str | None
    python_version: str
    platform_summary: str
    cpu_count: int | None
    total_ram_gb: float | None
    judge_model: str
    judge_model_digest: str | None
    embedding_model: str
    embedding_model_digest: str | None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _git_sha() -> str | None:
    return _run_git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    """True when the working tree has uncommitted changes (results are not tied to a commit)."""
    status = _run_git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)


def _uv_lock_sha256() -> str | None:
    """Hash the lockfile so dependency drift is visible across runs."""
    try:
        return hashlib.sha256(UV_LOCK_PATH.read_bytes()).hexdigest()
    except OSError:
        return None


def _total_ram_gb() -> float | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return round(page_size * page_count / 1024**3, 1)


def _ollama_model_digests(ollama_url: str) -> dict[str, str]:
    """Map model name -> digest from ``GET /api/tags``.

    A model tag like ``gemma3:4b`` is mutable — it can be re-pulled and point at
    different weights. The digest is what actually pins the model.
    """
    try:
        resp = httpx.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError):
        return {}

    digests: dict[str, str] = {}
    for model in body.get("models", []):
        name = model.get("name")
        digest = model.get("digest")
        if isinstance(name, str) and isinstance(digest, str):
            digests[name] = digest
    return digests


def _lookup_digest(digests: dict[str, str], model: str) -> str | None:
    """Look up a model digest, tolerating the implicit ``:latest`` tag."""
    if model in digests:
        return digests[model]
    if ":" not in model:
        return digests.get(f"{model}:latest")
    return None


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
    judge_model: str,
    embedding_model: str,
    ollama_url: str,
) -> RunMetadata:
    """Collect provenance for the current eval run. Never raises."""
    digests = _ollama_model_digests(ollama_url)
    return RunMetadata(
        seed=seed,
        git_sha=_git_sha(),
        git_dirty=_git_dirty(),
        uv_lock_sha256=_uv_lock_sha256(),
        python_version=sys.version.split()[0],
        platform_summary=platform.platform(),
        cpu_count=os.cpu_count(),
        total_ram_gb=_total_ram_gb(),
        judge_model=judge_model,
        judge_model_digest=_lookup_digest(digests, judge_model),
        embedding_model=embedding_model,
        embedding_model_digest=_lookup_digest(digests, embedding_model),
        settings_snapshot=_settings_snapshot(),
    )

"""Small, explicit live-local long-context benchmark adapter.

This module intentionally has no provider-specific retrieval magic: the corpus
is the dataset's ordered citations, lexical ranking is the fixed baseline, and
Ollama is only used for capability discovery and generation.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from evals.metrics import exact_match, f1


class CapabilityStatus(StrEnum):
    AVAILABLE = "available"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"


class ContextBudgetError(ValueError):
    """The prompt overhead and output reservation leave no input budget."""


@dataclass(frozen=True)
class ModelCapability:
    model: str
    status: CapabilityStatus
    native_context_window: int | None = None
    digest: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class WindowValidation:
    status: CapabilityStatus
    reason: str | None = None


@dataclass(frozen=True)
class ContextPlan:
    chunks: list[dict[str, Any]]
    strategy: str
    budget_tokens: int
    context_tokens: int
    input_tokens: int
    truncated: bool
    counting_scheme: str = "whitespace-v1"


def count_tokens(text: str) -> int:
    """Use a deterministic conservative estimate when no model tokenizer exists."""
    return len(text.split())


def _rank(question: str, chunk: dict[str, Any]) -> tuple[float, str]:
    terms = set(re.findall(r"\w+", question.casefold()))
    words = re.findall(r"\w+", str(chunk.get("text", "")).casefold())
    lexical = sum(word in terms for word in words) / max(len(words), 1)
    return (-float(chunk.get("score", lexical)), str(chunk.get("id", "")))


def build_context(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    window: int,
    prompt_overhead_tokens: int,
    output_tokens: int,
    strategy: str = "stuff",
    top_k: int = 5,
) -> ContextPlan:
    if strategy not in {"fixed_top_k", "stuff"}:
        raise ValueError("strategy must be fixed_top_k or stuff")
    budget = window - prompt_overhead_tokens - output_tokens
    if budget < 1:
        raise ContextBudgetError("context window is consumed by prompt overhead and output budget")
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in sorted(chunks, key=lambda item: _rank(question, item)):
        chunk_id = str(chunk.get("id", ""))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        ordered.append(dict(chunk))
    if strategy == "fixed_top_k":
        ordered = ordered[:top_k]
    selected: list[dict[str, Any]] = []
    used = 0
    truncated = False
    for chunk in ordered:
        text = str(chunk.get("text", ""))
        tokens = count_tokens(text)
        if used + tokens > budget:
            truncated = True
            if strategy == "stuff" and not selected:
                raise ContextBudgetError("first context chunk exceeds the available input budget")
            break
        selected.append(chunk)
        used += tokens
    return ContextPlan(selected, strategy, budget, used, prompt_overhead_tokens + used, truncated)


def probe_model_capability(client: Any, model: str) -> ModelCapability:
    """Probe Ollama's model metadata without assuming a context length."""
    try:
        response = client.post("/api/show", json={"name": model})
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        return ModelCapability(model, CapabilityStatus.UNAVAILABLE, reason=f"probe failed: {exc}")
    info = body.get("model_info", {})
    raw_window = next(
        (value for key, value in info.items() if key.endswith("context_length")), None
    )
    if not isinstance(raw_window, int) or raw_window < 1:
        return ModelCapability(
            model, CapabilityStatus.UNAVAILABLE, reason="Ollama did not advertise context_length"
        )
    digest = body.get("details", {}).get("digest") or body.get("digest")
    return ModelCapability(model, CapabilityStatus.AVAILABLE, raw_window, digest)


def validate_context_window(
    requested: int, *, native_context_window: int | None
) -> WindowValidation:
    if native_context_window is None:
        return WindowValidation(CapabilityStatus.UNAVAILABLE, "native context limit is unknown")
    if requested > native_context_window:
        return WindowValidation(
            CapabilityStatus.UNSUPPORTED,
            f"requested window {requested} exceeds native model limit {native_context_window}",
        )
    return WindowValidation(CapabilityStatus.AVAILABLE)


def corpus_identity(chunks: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{chunk.get('id', '')}\0{chunk.get('text', '')}" for chunk in chunks)
    return hashlib.sha256(payload.encode()).hexdigest()


def generate_live_answer(
    client: Any,
    *,
    model: str,
    question: str,
    context: ContextPlan,
    timeout_seconds: float,
    seed: int,
) -> tuple[str, dict[str, float | int]]:
    prompt = (
        "Answer using only the context.\n\n"
        + "\n\n".join(f"[{chunk['id']}] {chunk['text']}" for chunk in context.chunks)
        + f"\n\nQuestion: {question}"
    )
    started = time.perf_counter()
    response = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"seed": seed},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    body = response.json()
    answer = str(body.get("message", {}).get("content", "")).strip()
    elapsed = time.perf_counter() - started
    return answer, {
        "seconds": elapsed,
        "input_tokens": context.input_tokens,
        "output_tokens": count_tokens(answer),
    }


def make_live_executor(
    records: list[Any], *, base_url: str, model: str, seed: int, timeout: float
) -> Any:
    """Create a matrix executor; network work is only performed in live-local mode."""
    by_id = {record.record_id: record for record in records}
    client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
    capability = probe_model_capability(client, model)

    def execute(case: Any, _: Any) -> dict[str, Any]:
        config = case.effective_config
        requested = int(config["context_window"])
        validation = validate_context_window(
            requested, native_context_window=capability.native_context_window
        )
        model_info = {
            "provider": "ollama",
            "generation": model,
            "digest": capability.digest or "unknown",
        }
        if validation.status is not CapabilityStatus.AVAILABLE:
            return {
                "status": "unsupported"
                if validation.status is CapabilityStatus.UNSUPPORTED
                else "unavailable",
                "model": model_info,
                "error": {
                    "type": "CapabilityError",
                    "message": validation.reason or "capability unavailable",
                },
                "latency": {
                    "retrieval_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "scoring_seconds": 0.0,
                    "total_seconds": 0.0,
                },
                "resources": {
                    "cpu_rss": {"status": "unavailable", "reason": "not sampled in first slice"},
                    "gpu_vram": {"status": "unavailable", "reason": "not sampled in first slice"},
                    "warm_state": "unknown",
                },
            }
        case_results: list[dict[str, Any]] = []
        for record in by_id.values():
            chunks = [
                {"id": citation.citation_id, "text": citation.text} for citation in record.citations
            ]
            retrieval_started = time.perf_counter()
            plan = build_context(
                record.question,
                chunks,
                window=requested,
                prompt_overhead_tokens=32,
                output_tokens=256,
                strategy=str(config.get("context_strategy", "fixed_top_k")),
                top_k=int(config.get("top_k", 5)),
            )
            retrieval_seconds = time.perf_counter() - retrieval_started
            answer, generation = generate_live_answer(
                client,
                model=model,
                question=record.question,
                context=plan,
                timeout_seconds=timeout,
                seed=seed,
            )
            scoring_started = time.perf_counter()
            metrics = {
                "exact_match": exact_match(answer, record.reference_answers_or_default()),
                "f1": f1(answer, record.reference_answers_or_default()),
            }
            scoring_seconds = time.perf_counter() - scoring_started
            case_results.append(
                {
                    "record_id": record.record_id,
                    "status": "completed",
                    "metrics": metrics,
                    "latency": {
                        "retrieval_seconds": retrieval_seconds,
                        "generation_seconds": generation["seconds"],
                        "scoring_seconds": scoring_seconds,
                        "total_seconds": retrieval_seconds
                        + float(generation["seconds"])
                        + scoring_seconds,
                    },
                    "tokens": {
                        "context": plan.context_tokens,
                        "input": plan.input_tokens,
                        "output": generation["output_tokens"],
                        "counting_scheme": plan.counting_scheme,
                    },
                    "truncated": plan.truncated,
                    "corpus_checksum": corpus_identity(chunks),
                }
            )
        metric_names = ("exact_match", "f1")
        aggregate_metrics = {
            name: sum(float(item["metrics"][name]) for item in case_results) / len(case_results)
            for name in metric_names
        }
        aggregate_latency = {
            name: sum(float(item["latency"][name]) for item in case_results)
            for name in (
                "retrieval_seconds",
                "generation_seconds",
                "scoring_seconds",
                "total_seconds",
            )
        }
        return {
            "status": "completed",
            "model": model_info,
            "metrics": aggregate_metrics,
            "latency": aggregate_latency,
            "resources": {
                "cpu_rss": {"status": "unavailable", "reason": "not sampled in first slice"},
                "gpu_vram": {"status": "unavailable", "reason": "not sampled in first slice"},
                "warm_state": "unknown",
            },
            "effective_config": {
                **config,
                "native_context_window": capability.native_context_window,
                "record_results": case_results,
            },
        }

    return execute

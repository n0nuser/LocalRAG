"""Deterministic, provenance-preserving extractive context compression."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

COMPRESSOR_VERSION = "extractive-v1"
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)|[^\n]+$", re.UNICODE)
_FENCE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_TABLE_RE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\n?)+", re.MULTILINE)
_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


@dataclass(frozen=True)
class CompressionBudget:
    max_contexts: int = 5
    candidate_count: int = 20
    per_context_tokens: int = 256
    total_tokens: int = 1024
    per_context_chars: int = 4000
    total_chars: int = 16000


@dataclass(frozen=True)
class CompressionResult:
    contexts: list[dict[str, Any]]
    status: str
    compressor_version: str
    input_tokens: int
    output_tokens: int
    input_chars: int
    output_chars: int


def count_tokens(text: str) -> int:
    """Count whitespace-delimited tokens consistently for compression accounting."""
    return len(_TOKEN_RE.findall(text))


def compress_contexts(  # noqa: C901
    contexts: list[dict[str, Any]],
    question: str,
    budget: CompressionBudget,
    *,
    scorer: Callable[[str, str], float] | None = None,
) -> CompressionResult:
    """Compress ranked contexts without changing their source identity.

    Code fences and markdown tables are indivisible units. Any unit that cannot fit
    is omitted; this explicit no-context behavior is safer than emitting malformed
    code/table text or exceeding the hard budget.
    """
    source_texts = [_context_text(context) for context in contexts[: budget.candidate_count]]
    input_text = "\n\n".join(source_texts)
    if not contexts or not any(source_texts):
        return CompressionResult(
            contexts=[],
            status="no_context",
            compressor_version=COMPRESSOR_VERSION,
            input_tokens=count_tokens(input_text),
            output_tokens=0,
            input_chars=len(input_text),
            output_chars=0,
        )
    try:
        selected: list[dict[str, Any]] = []
        used_tokens = used_chars = 0
        for rank, original in enumerate(contexts[: budget.candidate_count]):
            if len(selected) >= budget.max_contexts:
                break
            text = _context_text(original)
            if not text:
                continue
            units = _units(text)
            scorer_fn = scorer or _lexical_score
            ranked = sorted(
                enumerate(units),
                key=lambda item: (-scorer_fn(question, item[1][0]), item[0]),
            )
            chosen: list[tuple[int, str, int, int]] = []
            context_tokens = context_chars = 0
            for index, (unit, start, end) in ranked:
                unit_tokens = count_tokens(unit)
                unit_chars = len(unit)
                if not unit.strip() or unit_tokens == 0:
                    continue
                if (
                    context_tokens + unit_tokens > budget.per_context_tokens
                    or context_chars + unit_chars + (1 if chosen else 0) > budget.per_context_chars
                    or used_tokens + unit_tokens > budget.total_tokens
                    or used_chars + context_chars + unit_chars + (1 if chosen else 0)
                    > budget.total_chars
                ):
                    continue
                chosen.append((index, unit, start, end))
                context_tokens += unit_tokens
                context_chars += unit_chars
            if not chosen:
                continue
            chosen.sort(key=lambda item: item[0])
            compressed = "\n".join(item[1].strip() for item in chosen)
            output = dict(original)
            output.pop("expanded_text", None)
            output["text"] = compressed
            output["compression"] = {
                "source": str(original.get("source", "unknown")),
                "chunk_index": int(original.get("chunk_index", -1)),
                "parent_id": _parent_id(original),
                "selected_spans": [
                    {"start": start, "end": end, "unit_index": index}
                    for index, _, start, end in chosen
                ],
                "original_rank": rank,
                "input_tokens": count_tokens(text),
                "output_tokens": context_tokens,
                "input_chars": len(text),
                "output_chars": len(compressed),
                "compressor_version": COMPRESSOR_VERSION,
                "status": "compressed",
            }
            selected.append(output)
            used_tokens += context_tokens
            used_chars += len(compressed)
        status = "compressed" if selected else "no_context"
        return CompressionResult(
            contexts=selected,
            status=status,
            output_tokens=used_tokens,
            output_chars=used_chars,
            compressor_version=COMPRESSOR_VERSION,
            input_tokens=count_tokens(input_text),
            input_chars=len(input_text),
        )
    except Exception:
        # A bounded extraction is preferable to failing a valid query. The same
        # hard limits apply, so failure can never silently overflow the prompt.
        fallback = compress_contexts(contexts, question, budget, scorer=lambda _q, _text: 0.0)
        fallback_contexts = []
        for context in fallback.contexts:
            output = dict(context)
            compression = dict(output.get("compression") or {})
            compression["status"] = "fallback"
            output["compression"] = compression
            fallback_contexts.append(output)
        return CompressionResult(
            contexts=fallback_contexts,
            status="fallback",
            compressor_version=COMPRESSOR_VERSION,
            input_tokens=fallback.input_tokens,
            output_tokens=fallback.output_tokens,
            input_chars=fallback.input_chars,
            output_chars=fallback.output_chars,
        )


def _context_text(context: dict[str, Any]) -> str:
    return str(context.get("expanded_text") or context.get("text") or "")


def _parent_id(context: dict[str, Any]) -> str | None:
    metadata = context.get("metadata") or {}
    value = metadata.get("parent_id") or metadata.get("heading_path")
    return str(value) if value else None


def _lexical_score(question: str, text: str) -> float:
    terms = set(re.findall(r"\w+", question.casefold(), re.UNICODE))
    if not terms:
        return 0.0
    words = re.findall(r"\w+", text.casefold(), re.UNICODE)
    return sum(word in terms for word in words) / max(1, len(words))


def _units(text: str) -> list[tuple[str, int, int]]:
    units: list[tuple[str, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for match in _FENCE_RE.finditer(text):
        units.append((match.group(), match.start(), match.end()))
        occupied.append((match.start(), match.end()))
    for match in _TABLE_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        units.append((match.group().strip(), match.start(), match.end()))
        occupied.append((match.start(), match.end()))
    for match in _SENTENCE_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        value = match.group().strip()
        if value:
            start = match.start() + (len(match.group()) - len(match.group().lstrip()))
            units.append((value, start, start + len(value)))
    return sorted(units, key=lambda item: item[1])

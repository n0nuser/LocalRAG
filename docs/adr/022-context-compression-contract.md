# ADR 022: Extractive Context Compression Contract

- Status: accepted
- Date: 2026-08-04

## Context

Retrieved chunks and optional parent-section expansion can consume the entire
generation prompt even when only a few sentences are relevant. Compression must
reduce prompt noise without changing retrieval scores or citation identity. This
is a prompt-stage decision, not a new retrieval or summarization source.

## Decision

When `CONTEXT_COMPRESSION_ENABLED=true`, the fixed pipeline is:

`retrieve -> optional rerank -> freshness/ranking -> parent expansion -> extractive compression -> prompt -> generation`.

The deterministic first backend uses Unicode-aware sentence units, complete fenced
code blocks, and complete contiguous Markdown table blocks. It scores units by
query-term overlap, breaks ties by original position, selects in retrieval-rank
order, and restores original order within each context. Code/table units that do
not fit are omitted rather than split. This also handles multilingual text without
language-specific tokenization.

The compressor uses one documented token approximation everywhere:
`len(re.findall(r"\\S+", text))`. Per-context and total token/character budgets
are hard limits. `CONTEXT_COMPRESSION_TOTAL_TOKENS` must fit inside
`LLM_CONTEXT_WINDOW_TOKENS` after the reserved prompt and answer budgets.

Each selected context retains its original `source`, `chunk_index`, metadata, and
optional parent identity. `compression` records the compressor version, original
rank, selected character spans/unit indexes, input/output token and character
counts, and status. Prompt text is replaced, but final source extraction still
uses the original retrieval identity; compression never fabricates citations.

Empty input, no fitting unit, duplicate/overlapping text, and oversized indivisible
units produce an explicit `no_context` result when nothing safe fits. Scorer,
tokenizer, or timeout failures use the same bounded deterministic selection with
`status="fallback"`; they never exceed a configured budget. An LLM/abstractive
summarizer is deliberately out of scope and remains a future isolated backend.

Compression is disabled by default, so existing prompt behavior and the RAGAS and
manual-only evaluation workflow remain unchanged.

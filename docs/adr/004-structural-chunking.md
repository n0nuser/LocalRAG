# ADR 004: Structural chunking with fixed fallback

## Context

The previous ingestion pipeline always split content with a fixed character window
(`chunk_chars` + overlap). That behavior can cut markdown sections, table rows,
and fenced code blocks in half. This increases retrieval noise and causes answer
errors even when the LLM is correct.

## Decision

Adopt structural chunking as the default:

- Markdown chunks respect headings and preserve table/code boundaries.
- Text/code files use paragraph/function-aware packing.
- Keep the old fixed-window chunker as an explicit fallback (`chunking_mode=fixed`).

Persist chunk metadata (`heading_path`, `chunk_type`) with each chunk to aid
retrieval inspection and downstream ranking.

## Consequences

- Better retrieval precision for structured documents without changing models.
- Slightly more ingestion logic and metadata surface area.
- Existing users can revert to fixed behavior with one setting if needed.

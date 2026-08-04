# ADR 021: Chunking Strategy Contract

- Status: accepted
- Date: 2026-08-04

## Context

Fixed and structural chunking historically returned different internal shapes,
while ingestion added provenance only after splitting. Additional strategies
need a stable seam without changing retrieval citations or structural
parent-section expansion.

## Decision

All strategies produce `localrag.ingestion.contract.Chunk` values. Each value
has non-empty normalized text, zero-based source order, optional source and
parent identifiers, strategy metadata, and a deterministic ID. IDs are SHA-256
of the contract version, source identity, strategy, logical index, and chunk
text. They do not depend on timestamps, process order, or object identity;
duplicate text remains distinct because its index differs.

Offsets are explicitly absent (`None`). Current strategies strip whitespace
and structural chunking repacks blocks, so offsets would imply precision they
do not provide. Empty input emits no chunks. An atomic value larger than the
configured limit is emitted intact with `metadata["oversized"]` set to true;
input is never silently dropped. Recursive splitting normalizes separator
whitespace when packing chunks.

The supported modes are `fixed`, `structural`, and `recursive`. Fixed and
structural metadata and behavior remain compatible. Recursive is the first
additional strategy and splits on paragraph, line, word, then character
boundaries, with configured overlap when packed chunks cross a boundary.

## Consequences

Vector-store IDs use the contract ID when present and retain the legacy
`source:index` fallback for older callers. Re-ingesting unchanged input with
the same settings produces the same IDs. Changing source content, strategy, or
contract version produces different IDs.

Semantic, sentence-window, and parent-child strategies are intentionally not
part of this slice. Sentence-window and parent-child behavior belong in future
retrieval/context-assembly decisions. Benchmark matrix expansion is also
deferred; strategy selection and the contract are the small fixture/protocol
seam for that future work.

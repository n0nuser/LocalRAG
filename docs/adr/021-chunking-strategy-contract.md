# ADR 021: Chunking Strategy Contract

- Status: accepted; amended 2026-10-03 (the chunk record became the single owner of chunk identity and metadata, and the vector store's legacy ID fallback was removed)
- Date: 2026-08-04

## Context

Fixed and structural chunking historically returned different internal shapes,
while ingestion added provenance only after splitting. Additional strategies
need a stable seam without changing retrieval citations or structural
parent-section expansion.

The amendment addresses a second problem: chunk metadata was an untyped dict whose keys were written by string literal in ingestion and read by string literal in about nine other modules, and chunk IDs had two derivation rules.
Every change to a chunk field had to touch all of them.

## Decision

The chunk record, `localrag/chunks/record.py`, is the only owner of three things: chunk identity, the stored metadata schema, and the retrieval-context shape.
Ingestion writes through it; storage, retrieval, prompt, compression, and citation code read through it.
No module outside `localrag/chunks/` reads or writes chunk metadata by string key.

Strategies (`fixed.py`, `structural.py`, `recursive.py` in `localrag/chunks/`) produce `ChunkDraft` values: non-empty normalized text, a heading path, a chunk type, and, for recursive chunking, an `oversized` flag.
`chunk_source` in `localrag/chunks/strategies.py` dispatches on `chunking_mode` and places each draft as a frozen `Chunk` with its source, zero-based index, strategy, and chunk ID.
A placed chunk always has an ID, so an ID-less chunk is unrepresentable.

IDs are SHA-256 of the contract version, source identity, strategy, logical index, and chunk text, derived only in `Chunk.place`.
They do not depend on timestamps, process order, or object identity; duplicate text remains distinct because its index differs.

`ChunkMetadata` is the stored row: the chunk's own fields plus the `SourceProvenance` shared by every chunk of a source (file type, ingestion time, content hash, modification time, git commit, tenant).
Writes are complete and omit `None` fields, because Chroma rejects a `None` value on `add` and silently drops the key on `upsert`.
Reads are tolerant: a missing, ill-typed, or unknown key reads as the field's default and never raises, because collections hold rows from older writers and retriever plugins return arbitrary metadata.
`RetrievalContext` (the dict a retriever returns, and the retriever plugin contract's return type) is defined beside it and built by `retrieval_context`.

The stored row format is fixed by the collections already on disk: the same keys, values, and Python types as before the record existed.
`tests/test_stored_chunk_format.py` pins it per strategy, and `tests/integration/test_chunk_record_round_trip.py` checks it through real Chroma into retrieval contexts.

Offsets are explicitly absent. Current strategies strip whitespace
and structural chunking repacks blocks, so offsets would imply precision they
do not provide. Empty input emits no chunks. An atomic value larger than the
configured limit is emitted intact with `oversized` set to true;
input is never silently dropped. Recursive splitting normalizes separator
whitespace when packing chunks.

The supported modes are `fixed`, `structural`, and `recursive`. Fixed and
structural metadata and behavior remain compatible. Recursive is the first
additional strategy and splits on paragraph, line, word, then character
boundaries, with configured overlap when packed chunks cross a boundary.

## Consequences

`VectorStore.replace_source` takes the chunk IDs explicitly and no longer derives any ID itself.
The legacy `source:index` fallback is gone, along with the non-atomic `add_chunks` writer whose only production caller relied on it.
Rows written under a fallback ID are still replaced correctly, because `replace_source` deletes every old ID of the source that the new version does not reuse.

Re-ingesting unchanged input with the same settings produces the same IDs. Changing source content, strategy, or
contract version produces different IDs.

Adding or changing a stored chunk field means changing `ChunkMetadata` and its write in `ChunkMetadata.of`, and it is a stored-format change: the pinning test fails until the new format is deliberately accepted.

Semantic, sentence-window, and parent-child strategies are intentionally not
part of this slice. Sentence-window and parent-child behavior belong in future
retrieval/context-assembly decisions. Benchmark matrix expansion is also
deferred; strategy selection and the contract are the small fixture/protocol
seam for that future work.

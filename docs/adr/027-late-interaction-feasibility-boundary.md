# ADR 027: Late-interaction feasibility boundary

## Context

LocalRAG currently retrieves one dense vector per chunk from Chroma and can
fuse it with BM25. ColBERT-style retrieval stores a matrix of contextual token
vectors per chunk and scores each query token by its maximum document-token
similarity (MaxSim). That changes storage, indexing, query compute, and model
dependencies; it must not silently introduce incompatible Chroma vectors.

## Decision

Run late interaction as a research-only, offline comparison. The checked-in
prototype in `evals/late_interaction.py` accepts precomputed token matrices,
implements masked sum-of-maxima with optional query-token normalization, and
persists an explicit JSON artifact. It is not an embedding provider, is not
wired into dependency injection, and is not a Chroma backend.

The real-model follow-up, if approved, must use `colbert-ir/colbertv2.0` at a
recorded immutable Hugging Face revision, its model tokenizer, float32
embeddings, and a versioned LocalRAG corpus/query snapshot. The model card and
ColBERT repository are MIT licensed. `torch`, `transformers`, and the ColBERT
runtime remain optional research dependencies; they must not be added to the
default install without a separate decision. Downloads and environment setup
are preparation, never benchmark execution. CPU indexing/search must be
measured explicitly; a GPU is an optimization, not an assumption.

## Evidence boundary

`research/70-late-interaction-spike/fixture.json` is a deterministic four-query,
four-chunk smoke fixture with explicit vectors. The runner emits the existing
#84 `ResultFile` shape and #73 matrix-manifest shape. It proves MaxSim,
masking, persistence, deterministic ordering, and measurement plumbing only.
It is intentionally not evidence of model quality. RAGAS and manual-only
evaluation workflows are unchanged.

## Adoption matrix

| Criterion | Adopt threshold | Outcome required before adoption |
| --- | --- | --- |
| Quality | NDCG@10 improves at least 5 percentage points over hybrid on the same annotated corpus | Real-model result |
| Latency | Warm p95 no more than 2x hybrid and cold p95 is reported | CPU and GPU results |
| Memory | Peak RAM no more than 2x hybrid; VRAM is reported when used | Resource samples |
| Storage | Serialized token index no more than 5x dense index, or justified quality tradeoff | Artifact byte counts |
| Dependency/license | Offline install works; licenses and model terms are acceptable | Locked environment and inventory |
| Maintenance | Isolated adapter, migration plan, and owner exist | Follow-up design issue |

All criteria are gates, not a weighted average. The current evidence does not
meet the quality, resource, or real-model gates, so the decision is **reject for
default adoption; defer** a real-model benchmark until a representative
annotated corpus and hardware are available. No production integration follows
from this spike.

## Consequences

The default provider/retriever and Chroma collection remain unchanged. The
fixture can be run offline on CPU with no downloads. A future benchmark may add
optional dependencies and a real index only behind an explicitly approved
follow-up.

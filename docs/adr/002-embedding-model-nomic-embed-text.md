# ADR 002: Default Embedding Model — nomic-embed-text

**Status:** Accepted  
**Date:** 2026-04-30

## Context

The ingestion and retrieval pipeline requires an embedding model to convert text chunks and queries into vectors. The model must be:

1. Available locally via Ollama (offline-first constraint).
2. Produce high-quality semantic representations for English technical documentation.
3. Small enough to run on a CPU-only laptop within a reasonable time budget.

Candidates evaluated:

| Model | Dims | Approx size | Notes |
| --- | --- | --- | --- |
| **nomic-embed-text** | 768 | ~274 MB | Strong MTEB scores; ships with Ollama; no GPU needed |
| mxbai-embed-large | 1024 | ~670 MB | Higher quality but significantly larger |
| all-minilm | 384 | ~46 MB | Very small; quality drops noticeably on technical content |
| text-embedding-3-small (OpenAI) | 1536 | cloud | Excellent quality but breaks offline constraint |

## Decision

Default to **`nomic-embed-text`** via `OLLAMA_EMBED_MODEL`. Users may override per request or globally via `.env`.

## Rationale

- Bundled with Ollama and available with a single `ollama pull nomic-embed-text`.
- Strong performance on retrieval benchmarks relative to its size.
- 768-dimensional vectors keep ChromaDB storage overhead small for most document corpora.
- Override mechanism (`embed_model` field on ingest/query endpoints) allows teams to upgrade without code changes.

## Consequences

- **Positive**: works out of the box on any machine that can run Ollama.
- **Negative**: quality ceiling below cloud-native models like `text-embedding-3-large`. Teams with GPU resources can switch to `mxbai-embed-large` or a cloud model by updating `OLLAMA_EMBED_MODEL`.
- **Collection compatibility**: changing the embedding model for an existing collection requires a `POST /collections/rebuild` to re-embed all stored chunks.

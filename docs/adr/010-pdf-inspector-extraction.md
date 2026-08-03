# ADR 010: PDF text extraction with pdf-inspector

## Context

PDF ingestion extracted text with `pypdf`'s `extract_text()`, which returns a flat
character dump. It carries no layout information, so multi-column pages come out
interleaved, tables lose their row/column structure, and headings are
indistinguishable from body text. For a RAG pipeline this degrades chunk quality
at the source: no later stage can recover structure that extraction discarded.

Scanned pages were identified with a proxy heuristic — a page whose extracted text
was shorter than `OCR_MIN_CHARS_PER_PAGE` was assumed to be scanned and sent to
OCR. That test cannot distinguish a genuinely blank page from a page whose text
layer exists but decodes to garbage through a broken CID font encoding. The first
wastes an OCR pass; the second silently ingests corrupt text.

## Decision

Use [`pdf-inspector`](https://github.com/firecrawl/pdf-inspector) (Rust core with
PyO3 bindings, MIT, by Firecrawl) for PDF text-layer extraction, via
`extract_pages_markdown()`, which returns per-page Markdown plus layout metadata.

Retain the existing `pypdfium2` + Tesseract OCR fallback unchanged.
`pdf-inspector` performs no OCR by design — it classifies pages and reports which
ones lack usable text (`PageMarkdown.needs_ocr`, with `markdown` empty for those
pages), leaving the caller to route them. It is a triage layer, not an OCR
replacement, so rasterization plus Tesseract remains the only component able to
turn pixels into characters.

OCR now triggers when `page.needs_ocr` is set **or** the page's Markdown is
shorter than `ocr_min_chars_per_page`. The library's per-page signal derives from
content-stream and font-encoding analysis rather than character count; the length
check is kept so the existing setting stays meaningful and short-page behavior
remains backward compatible.

`pypdf` is dropped from dependencies — nothing else in the repo used it.

## Reported performance

From the upstream [opendataloader-bench](https://github.com/opendataloader-project/opendataloader-bench)
results (200 PDFs, local engines without model-based parsing, OCR disabled,
refreshed 2026-07-31 on an Apple M4 Pro). Scores 0–1, higher is better:

| Engine | Overall | Reading order | Tables (TEDS) | Headings | Speed (200 docs) |
|---|---|---|---|---|---|
| **pdf-inspector** | **0.875** | **0.915** | **0.814** | 0.788 | **0.470s** |
| liteparse | 0.873 | 0.913 | 0.693 | **0.811** | 0.750s |
| opendataloader | 0.831 | 0.902 | 0.489 | 0.739 | 2.569s |
| pymupdf4llm | 0.735 | 0.886 | 0.401 | 0.424 | 17.117s |
| markitdown | 0.589 | 0.844 | 0.273 | 0.000 | 16.165s |

These are **vendor-published figures on a corpus of native-text PDFs, not
measurements taken against LocalRAG's own documents.** `pypdf` — what we actually
replaced — does not appear in the comparison, so the table establishes that
pdf-inspector is competitive among local Markdown extractors, not a quantified
speedup for this codebase. The upstream benchmark states its best fit as
native-text PDFs; every engine ran with OCR disabled, so it says nothing about
scanned-document behavior, which is precisely the path we left untouched.

Locally verified: extraction over a real 2-page BOCyL government notice produced
genuine Markdown structure (H1–H4 headings, bold/italic emphasis, links) where
`pypdf` previously produced flat text.

## Consequences

- Native-text PDFs ingest as Markdown, preserving headings, tables, lists, and
  multi-column reading order.
- Scanned-PDF behavior is unchanged; the OCR path and its memory bounds for
  oversized architectural sheets are preserved verbatim.
- OCR routing keys off a font-encoding signal instead of a character count,
  catching broken CID encodings the length heuristic silently accepted.
- A compiled dependency enters the stack. Prebuilt `abi3` wheels cover
  manylinux x86_64/aarch64, macOS, and Windows; `python:3.13-slim` is covered, so
  no Rust toolchain is needed in the Docker build. Platforms without a wheel
  would build from source.
- Heading detection is heuristic (font-size ratios), so unusual typography can
  produce spurious heading levels.
- `chunk_document()` dispatches on file extension, and `.pdf` is not in
  `MARKDOWN_EXTENSIONS`, so PDF chunks still take the non-Markdown path and carry
  an empty `heading_path`. Routing PDFs to the Markdown chunker to realize the
  retrieval benefit of ADR 004 is deliberately left as follow-up work, since it
  warrants its own before/after validation on real documents.

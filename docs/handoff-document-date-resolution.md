# Handoff: document date resolution + RAGAS validation

Two pieces of work, in order. The second validates the first plus what already
shipped.

## Context: what already landed

Four PRs merged to `main` (all present as of `afede21`):

| PR | What |
| --- | --- |
| #51 | PDF text extraction moved from `pypdf` to `pdf-inspector` (Rust). 89.0ms → 10.1ms on a real 2-page PDF, 6% more text |
| #52 | Lazy OCR document open (14.6ms → 11.2ms); PDFs chunk by heading structure (3 flat chunks → 9 with hierarchical `heading_path`) |
| #53 | `docs/research-pipeline-performance.md` — 10 ranked pipeline findings |
| #54 | Recency reworked from a score multiplier into an RRF ranked list — see [ADR 006](adr/006-freshness-decay.md) amendment |

Read the ADR 006 amendment before touching anything here. It explains why recency
is a *rank* contribution and not a multiplier, with the arithmetic.

---

## Task 1: three-tier document date resolution

### Why

Recency ranking currently keys on `ingested_at`, which is
`datetime.now(UTC).isoformat()` captured at ingest time
(`localrag/ingestion/service.py:221`). That is **wall-clock upload time, not the
document's own date**. A bulk re-ingest resets it for the entire corpus
simultaneously, erasing the signal recency ranking depends on.

### Required behavior

Resolve a chunk's date from the first trustworthy source available, in this
order:

1. **Date parsed from the document's own text.** Preferred. A PDF that says
   "15 de febrero de 2022" should rank as a 2022 document regardless of when it
   was uploaded. Only accept when parsing is confident — see the trust rules
   below.
2. **Document metadata.** PDF `/CreationDate` / `/ModDate`, docx core properties,
   and `source_mtime` (already captured at `service.py:223` and stored in chunk
   metadata, currently unused for ranking).
3. **`ingested_at`.** Existing behavior, as the final fallback.

Store the resolved value in a **new** metadata field (suggest `document_date`)
plus a `document_date_source` field recording which tier won — the second field
matters for debugging and for the RAGAS comparison below. Do **not** overwrite
`ingested_at`; it stays as the audit trail of when ingestion happened.

### Trust rules — the part that needs judgement

"If not present and trustworthy" was the explicit instruction, so in-text parsing
must be conservative. A wrong date is worse than no date, because it silently
reorders results. Suggested constraints, to be confirmed by whoever implements:

- Reject dates in the future, and dates implausibly far in the past.
- Prefer dates near the start of the document (headers, mastheads, dateline)
  over the first date appearing anywhere — a date inside body prose is often a
  reference to some other event, not the document's own date.
- Be careful with locale: the corpus in `data/uploads/` is Spanish
  (`Núm. 31 Martes, 15 de febrero de 2022`). Do not assume English month names or
  US `MM/DD` ordering.
- When two tiers disagree wildly, prefer the lower tier and log it. Consider a
  setting to disable in-text parsing entirely.

### Wiring

The retriever already funnels every date read through one helper —
`_parse_ingested_at` in `localrag/rag/retriever.py:22`. Three callers depend on
it: the relevance tie-break (`_sorted_by_score`, `:203`), the recency ranking
(`_recency_ranks`, `:216`), and the vector-mode decay (`apply_freshness`, `:242`).
Prefer resolving the date at **ingest** time into metadata and having the
retriever read the new field, rather than parsing text at query time.

Touch points:

- `localrag/ingestion/service.py:221-238` — where `ingested_at`, `source_mtime`
  and `git_commit` are built into the metadata dict.
- `localrag/ingestion/parsers/*` — parsers may need to surface document metadata
  they currently discard. `pdf_inspector` exposes `PdfResult.title`; check
  whether the shipped `__init__.pyi` exposes creation dates before assuming.
- `localrag/rag/retriever.py:22` — the single parse helper.
- `.env.example`, settings docstring, `docs/rag-retrieval.md`, ADR 006 — the
  amendment already flags this as follow-up; update it when done.

### Verification expectations

Existing tests must keep passing (`166 passed, 3 skipped` at `afede21`). Follow
the pattern used in this work: write the test, then **mutation-test it** — revert
the fix and confirm the test actually fails. Two bugs in the recency work were
caught exactly this way, and one mutation initially passed because the tests
exercised a helper rather than the real `retrieve()` path.

Cover at minimum: each tier winning in turn; an untrustworthy in-text date being
rejected in favour of a lower tier; and a document with no date anywhere.

---

## Task 2: run RAGAS locally and evaluate

### Why

Everything above changed **what text gets retrieved and in what order**, and none
of it has been validated against relevance judgements. Two numbers in particular
are reasoned, not measured:

- `FRESHNESS_WEIGHT=0.15` (ADR 006 amendment) — derived from RRF gap arithmetic,
  never fitted to labelled data.
- The repeated-heading demotion threshold in `parse_pdf` (>50% of pages, minimum
  3 pages) — reasoned from one real document.

### How to run

```bash
# Offline mode — what CI uses; no live API needed
uv run python evals/run_evals.py --offline

# Live mode — exercises the real retrieval path, which is what actually matters here
uv run python evals/run_evals.py --api-url http://localhost:8000
```

The judge LLM and embeddings run on local Ollama via its OpenAI-compatible `/v1`
endpoint (default judge `gemma3:4b`, override with `--judge-model`). No external
API key. Results land in `evals/results/<timestamp>.json`.

**Prefer live mode for this evaluation.** `--offline` uses stored contexts from
`evals/dataset.json`, so it cannot see retrieval-order changes — and retrieval
order is precisely what PR #52 and #54 altered. Offline mode is the CI gate, not
a validation of this work.

Requires the Docker stack up (`docker compose up`). Note PR #49 (open, stale)
enables GPU passthrough for the Ollama container; without it Ollama runs
CPU-bound and a full eval took 9+ minutes and did not complete. Consider
rebasing and merging #49 first.

### What to compare

Capture a baseline **before** implementing Task 1, so the date work can be judged
separately from what already shipped:

1. `main` at `afede21` — everything above merged, `ingested_at` still the date
   source.
2. `FRESHNESS_WEIGHT=0` on the same commit — isolates how much recency is
   contributing at all.
3. After Task 1 — three-tier dates in place.

If (1) does not beat (2) on the retrieval metrics, that is a real finding and
should be reported plainly: it would mean recency is not earning its weight and
`FRESHNESS_WEIGHT` should drop toward 0. Do not tune the number to make the fix
look good.

The dataset is 23 items (`evals/dataset.json`) — small enough that
single-question differences move the aggregate. Report per-question deltas, not
just the mean, and treat small aggregate movements as noise.

---

## Repo conventions that apply

- **Trunk-based git.** Short-lived `feat/…` or `fix/…` off an updated `main`,
  PR to `main`, rebase (squash is disabled on this repo). Full policy in
  `.github/CONTRIBUTING.md`.
- **After each merged PR:** return to `main`, pull, delete the merged branch
  locally and remotely, `git remote prune origin`.
- **Docs in the same PR** as the change — see the table in `CLAUDE.md`. ADR 006
  and `docs/rag-retrieval.md` both describe date handling and will need updating.
- **Verification gates:** `uv run pytest tests/ -q`, `uv run ruff check`,
  `uv run ruff format --check`, `uv run mypy localrag/`. mypy has a **pre-existing
  baseline of 4 errors** (missing stubs for `sentence_transformers`, `pypdfium2`,
  `pytesseract`, `rank_bm25`) — confirm any mypy output matches that baseline
  rather than assuming a regression.

## Open branches

- **PR #49** `feat/ollama-gpu-passthrough` — stale, CI cancelled, behind `main`.
  Needs a rebase. Worth doing before the RAGAS run.
- **PRs #33–#39** — six dependabot PRs. Per the user's stated preference: regen
  the lock once, push directly, let dependabot auto-close the redundant ones.

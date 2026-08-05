# ADR 037 — Grouped configuration model behind flat public names

- **Status:** accepted
- **Date:** 2026-08-06
- **Amends:** [ADR 020](020-structured-configuration.md) (resolution order is unchanged)
- **Issue:** [#143](https://github.com/n0nuser/LocalRAG/issues/143) (Finding 2)

## Context

`Settings` was a single flat namespace of 116 fields whose
`validate_configuration` had grown to ~105 lines carrying
`# noqa: C901, PLR0912, PLR0915`. The fields already clustered by name prefix
(`adaptive_`, `hyde_`, `otel_`, …), and the YAML schema exposed nested sections
that were hand-flattened back onto the namespace — so a grouped model existed
conceptually but not in code, and the two surfaces had drifted.

The binding constraint is that the flat names are a **public contract**:
`.env.example`, `docs/configuration.md`, every deployment, and 124 call sites of
the form `settings.hyde_enabled` depend on them. Issue #143 states that no change
may silently drop a documented setting or env var.

Plain Pydantic nesting does exactly that. With `hyde: HydeSettings`, pydantic
expects `HYDE__ENABLED`; the documented `HYDE_ENABLED` stops resolving **silently**,
with no error and no warning — the setting simply reverts to its default. This
was verified before choosing the design.

## Decision

The model is grouped; the public surface stays flat.

1. **Grouped sub-models** live in `localrag/settings_groups.py`, one per bounded
   feature, each owning **its own validation**. `Settings` composes them
   (`ollama`, `chroma`, `chunking`, `embedding`, `ingest`, `upload`, `audit`,
   `ocr`, `retrieval`, `query_cache`, `api`, `llm`, `observability`), with
   `retrieval` further composing `expansion`, `hyde`, `rerank`, `compression`,
   and `adaptive`. `log_level`, `tenant_id`, and `eval_seed` stay top-level:
   they belong to no feature.

2. **`localrag/settings_map.py`** holds `FLAT_TO_PATH`, the one table mapping
   each public flat name to its dotted grouped path. It is the compatibility
   contract, and a test asserts it is **total in both directions** — a new
   grouped field cannot be added without a flat name, and a flat name cannot be
   dropped silently.

3. **`_FlatEnvSource` / `_FlatDotenvSource`** replace pydantic's env and dotenv
   sources, regrouping flat names before validation. Resolution order is
   unchanged from ADR 020: `--set` → env → `.env` → YAML → file secrets.

4. **Declared `@property` accessors** expose every flat name on `Settings`, so
   all 124 call sites are untouched. They are written out explicitly rather than
   attached dynamically, because mypy cannot see `setattr`-attached members —
   dynamic attachment produced 171 type errors.

5. **`with_overrides(**flat)`** replaces `model_copy(update={...})` for flat
   names. Because the flat names are properties rather than fields,
   `model_copy` would accept them and **silently do nothing**; three shipped
   call sites had that hazard and now use the explicit helper.

## Snapshot shape

`resolved_snapshot()` now mirrors the grouped model. `flat_snapshot()` returns
the same provenance keyed by flat names for consumers that want it.

`evals/environment.py` reads settings via `getattr(settings, flat_name)`, which
the flat properties satisfy unchanged, so `SNAPSHOT_SETTINGS_FIELDS` and the
`evals/results/*.json` shape are **unaffected** and need no schema bump. Secret
and host-path redaction applies to both projections.

## Consequences

- `validate_configuration` is gone; validation lives with the fields it
  constrains, and the complexity `noqa` is removed.
- Every documented env var, YAML key, and `--set` override keeps working.
  Parametrised tests assert this for a representative sample.
- `localrag config-show` output is now grouped. This is a visible change for
  anyone parsing that JSON; `flat_snapshot()` is the stable flat alternative.
- Adding a setting is now a three-step change: field on a group model, entry in
  `FLAT_TO_PATH`, and a flat property. The totality test fails loudly if any
  step is missed.

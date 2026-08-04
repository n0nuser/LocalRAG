# ADR 020: Structured Configuration Sources

**Status:** Accepted  
**Date:** 2026-08-04

## Context

The flat environment contract has become difficult to inspect as retrieval,
providers, and evaluation options grow. Existing `.env` deployments must keep
working, while API workers and CLI commands need the same validated configuration.

## Decision

Use plain YAML parsed into the existing Pydantic Settings model. There is no
repository need for Hydra's composition, sweep, or runtime plugin features.
The exact precedence is built-in defaults, YAML, `.env`, process environment,
then explicit CLI `--set FIELD=VALUE` overrides. `--config PATH` selects YAML
for CLI execution; `LOCALRAG_CONFIG` selects it for API startup. The API resolves
it in lifespan before cached dependency workers are built.

YAML has strict structured sections for `embedding`, `retrieval`, `generation`,
`dataset`, and `evaluation`. They map to the one flat `Settings` object consumed
by the application, so domain code does not receive competing configuration
models. Unknown YAML keys and CLI fields are errors. `${ENV_NAME}` interpolation
is supported, and YAML-relative paths resolve beside the YAML file. Without YAML,
existing relative `.env` paths retain current-working-directory behavior.

Existing environment variable names remain canonical compatibility names. The
legacy YAML `ollama.embed_model` and `ollama.base_url` aliases emit a
`DeprecationWarning` and may be removed after the next major release. Secrets
are supplied by environment or an untracked secret source only. Resolved
snapshots redact secret fields and replace filesystem paths with `<path>`.

## Consequences

Configuration is inspectable and deterministic without adding a Hydra runtime.
The snapshot is safe for benchmark provenance used by future evaluation work,
but it intentionally excludes credentials, host paths, and document content.
Users migrating from `.env` can adopt YAML incrementally; later sources continue
to override it.

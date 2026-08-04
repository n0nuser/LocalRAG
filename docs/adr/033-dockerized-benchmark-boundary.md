# ADR 033: Dockerized Benchmark Boundary

- **Status:** Accepted
- **Date:** 2026-08-04

## Context

Benchmark execution needs a repeatable integration environment without creating
a second dataset, selection, matrix, result, comparison, or failure-analysis
implementation. Containers reduce host drift, but local model inference can
still vary by hardware, drivers, kernels, quantization, and Ollama behavior.

## Decision

`docker-compose.benchmark.yml` is the dedicated integration boundary. The
benchmark image contains `evals/`, registered fixtures, the application source,
`scripts/`, `uv.lock`, and the committed model lock. It excludes secrets,
credentials, host documents, and host data. The CPU and GPU profiles use pinned
Ollama and Chroma image manifest digests, explicit CPU/memory/PID limits, named
isolated service volumes, and a bind-mounted result export.

The smoke profile has no network and runs the existing matrix runner against the
registered fixture. Its adapter emits the existing versioned `ResultFile`; it
does not replace dataset selection, matrix expansion, provenance, comparison, or
failure-artifact logic. Real profiles require an explicit `model-prep` step that
checks Ollama readiness and exact model content digests from `docker/models.lock.json`.
Preparation may be performed separately while network access is allowed; the
benchmark command never pulls models implicitly.

## Reproducibility claim

Same-host fixture runs with the same image, lockfiles, fixture checksum, profile,
and seed have identical selected IDs, ordering, schema-valid metadata, and
comparator decisions. Real model runs record the existing provenance and model
digests, but bit-for-bit text or score equality across hardware is explicitly not
promised. GPU support is opt-in and fails if the Docker runtime cannot provide
the requested NVIDIA device.

## Consequences

`docker compose ... --profile smoke run --rm benchmark-smoke` is a small CI
smoke path that downloads no large model. CPU/GPU runs are manual and leave
canonical results and per-case matrix artifacts below `evals/results/docker/`.
`down --volumes` resets isolated state; the entrypoint also removes stale result
exports before each run. RAGAS remains manual-only and no workflow invokes it.

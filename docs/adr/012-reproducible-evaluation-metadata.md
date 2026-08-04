# ADR 012: Reproducible evaluation metadata

## Context

"Reproducible" can mean reproducing the inputs and configuration or producing
identical model text. Local model output also depends on hardware, drivers,
quantization, runtime versions, and provider seeding behavior, so these claims
must not be conflated.

## Decision

Result metadata guarantees input/config reproducibility: dataset identity,
checksum, selected IDs, resolved seed and seed source, settings, code revision,
dependency lock hash, tracked package versions, Python/platform, CPU/RAM/GPU,
provider/model names, embedding identity, and model digests are recorded.
Seed coverage states which operations are controlled. Model tags are not
immutable identities; Ollama digests are recorded and are unavailable, rather
than unsupported, when a pulled-model digest cannot be read.

Probe values use capability envelopes with `available`, `unsupported`, or
`unavailable` status and a reason. Settings metadata is an explicit allowlist of
evaluation-affecting fields. Secrets and host-specific paths are excluded and
must not be added to the allowlist.

The contract does not promise bit-for-bit model-output determinism. Temperature
and seeds can reduce variance, but GPU arithmetic, Ollama/provider behavior,
model re-pulls, and live corpus state remain known limits.

## Consequences

- A result explains configuration and environment drift without dumping secrets.
- Missing probes remain visible instead of being confused with inapplicable
  capabilities.
- Comparisons must inspect model digests and provenance, not only model tags.
- Judge-backed changes require tolerance-based interpretation rather than an
  expectation of exact output equality.

## Related

[Reproducibility](../reproducibility.md), [ADR 011](011-evaluation-dataset-contract.md), [ADR 016](016-bounded-parallel-evaluation.md)

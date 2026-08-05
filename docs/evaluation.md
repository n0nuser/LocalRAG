# Evaluation

Start here for anything about measuring LocalRAG: datasets, metrics, run
reproducibility, benchmark reports, and the published leaderboard. Each section
below states what the topic covers and what it guarantees, then links to the
detailed page.

The harness lives in [`evals/`](../evals/) and ships in the wheel alongside
`localrag/`, because the `benchmark`, `report`, `leaderboard`, `eval`, and
`eval-compare` CLI commands import it.

## The pipeline in order

```
dataset  →  run  →  metrics  →  result JSON  →  report / comparison  →  leaderboard
```

| Stage | What it fixes | Detail |
| --- | --- | --- |
| **Dataset** | Which records are evaluated, and their identity/versioning | [eval-datasets.md](eval-datasets.md) |
| **Run** | Seeding, provenance, and what "reproducible" does and does not mean | [reproducibility.md](reproducibility.md) |
| **Metrics** | How scores are computed and bounded | [evaluation-metrics.md](evaluation-metrics.md) |
| **Reports** | Offline HTML benchmark output | [evaluation-reports.md](evaluation-reports.md) |
| **Leaderboard** | Publication rules and their strictness | [benchmark-leaderboard.md](benchmark-leaderboard.md) |

## The one thing to know first

Reproducibility here means **input/config reproducibility**, not bit-for-bit
model output. Given the same dataset version, seed, and configuration, a run
reproduces its *inputs and selection* exactly; local model text still depends on
hardware, drivers, quantization, and runtime versions. Claims that ignore this
distinction are the most common way benchmark numbers get misread — see
[reproducibility.md](reproducibility.md#reproducibility-levels) for the
precise levels.

## Commands

```bash
uv run localrag eval --offline          # deterministic fixture evaluation
uv run localrag benchmark               # canonical matrix run
uv run localrag report                  # offline HTML report
uv run localrag leaderboard             # publication (strict; refuses partial provenance)
uv run localrag eval-compare            # compare two result documents
```

## Decisions behind this

Evaluation is the most ADR-dense area of the repo, because each contract has to
survive schema evolution and stay comparable across runs:

| ADR | Contract |
| --- | --- |
| [011](adr/011-evaluation-dataset-contract.md) | Versioned dataset contract |
| [012](adr/012-reproducible-evaluation-metadata.md) | Reproducible run metadata |
| [013](adr/013-versioned-benchmark-results.md) | Versioned result documents |
| [014](adr/014-evaluation-metric-contract.md) | Metric contract |
| [015](adr/015-canonical-benchmark-matrix.md) | Canonical matrix runner |
| [016](adr/016-bounded-parallel-evaluation.md) | Bounded parallelism |
| [017](adr/017-strict-leaderboard-publication.md) | Strict publication rules |
| [009](adr/009-offline-ragas-judge.md) | RAGAS judge runs locally, not on OpenAI |
| [031](adr/031-failure-analysis-contract.md) | Per-case failure analysis |
| [018](adr/018-optional-mlflow-experiment-tracking.md), [026](adr/026-long-context-benchmark-boundary.md), [033](adr/033-dockerized-benchmark-boundary.md) | Optional tracking and benchmark boundaries |

The full status index is in [docs/adr/README.md](adr/README.md).

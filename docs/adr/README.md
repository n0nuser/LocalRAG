# Architecture Decision Records

ADRs are **immutable records of a decision at a point in time**. They are not
maintained documentation: when a decision changes, a new ADR supersedes the old
one and the old one stays as written. If you want current behavior, read the
docs in [`docs/`](../) — this index exists so you can tell, at a glance, which
records still bind and which are history.

Per [CONTRIBUTING](../../CONTRIBUTING.md), a new ADR is **required** when you set
or move a public extension boundary, a trust boundary, or a behavior-changing
default.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Accepted** | The decision is in force and describes current behavior. |
| **Amended** | Still in force, but a later ADR changed part of it. The amendment is noted at the top of the file. |
| **Research boundary** | Deliberately scopes something as research-only or not-yet-adopted. Binding as a *limit*, not as shipped behavior. |

## Index

| # | Decision | Status | Area |
| --- | --- | --- | --- |
| [001](001-vector-store-chromadb.md) | Vector store — ChromaDB | Accepted | Storage |
| [002](002-embedding-model-nomic-embed-text.md) | Default embedding model — nomic-embed-text | Accepted | Embedding |
| [003](003-agent-framework-anthropic-tool-use.md) | Agent framework — Anthropic tool use | Accepted | Agent |
| [004](004-structural-chunking.md) | Structural chunking with fixed fallback | Accepted | Ingestion |
| [005](005-hybrid-retrieval.md) | Hybrid retrieval (vector + BM25) | Accepted | Retrieval |
| [006](006-freshness-decay.md) | Freshness-aware retrieval scoring | Accepted | Retrieval |
| [007](007-resilient-llm-provider-routing.md) | Resilient, uniformly-routed LLM provider abstraction | Accepted | LLM |
| [008](008-cross-encoder-reranking.md) | Optional cross-encoder reranking | Accepted | Retrieval |
| [009](009-offline-ragas-judge.md) | RAGAS judge runs on local Ollama, not OpenAI | Accepted | Evaluation |
| [010](010-pdf-inspector-extraction.md) | PDF text extraction with pdf-inspector | Accepted | Ingestion |
| [011](011-evaluation-dataset-contract.md) | Versioned evaluation dataset contract | Accepted | Evaluation |
| [012](012-reproducible-evaluation-metadata.md) | Reproducible evaluation metadata | Accepted | Evaluation |
| [013](013-versioned-benchmark-results.md) | Versioned benchmark result documents | Accepted | Evaluation |
| [014](014-evaluation-metric-contract.md) | Evaluation metric contract | Accepted | Evaluation |
| [015](015-canonical-benchmark-matrix.md) | Canonical benchmark matrix runner | Accepted | Evaluation |
| [016](016-bounded-parallel-evaluation.md) | Bounded parallel evaluation | Accepted | Evaluation |
| [017](017-strict-leaderboard-publication.md) | Strict leaderboard publication | Accepted | Evaluation |
| [018](018-optional-mlflow-experiment-tracking.md) | Optional MLflow experiment tracking | Accepted | Evaluation |
| [019](019-embedding-provider-contract.md) | Embedding provider contract and collection compatibility | Accepted | Embedding |
| [020](020-structured-configuration.md) | Structured configuration sources | Amended by [037](037-grouped-configuration-model.md) | Configuration |
| [021](021-chunking-strategy-contract.md) | Chunking strategy contract | Accepted | Ingestion |
| [022](022-context-compression-contract.md) | Extractive context compression contract | Amended by [036](036-retire-zero-cost-feature-flags.md) | Retrieval |
| [023](023-bounded-adaptive-retrieval.md) | Bounded adaptive retrieval | Accepted | Retrieval |
| [024](024-embedding-cache-contract.md) | Provider-aware ingestion embedding cache | Amended by [036](036-retire-zero-cost-feature-flags.md) | Embedding |
| [025](025-hyde-retrieval-experiment.md) | Bounded HyDE retrieval experiment | Accepted (disabled by default) | Retrieval |
| [026](026-long-context-benchmark-boundary.md) | Long-context benchmark boundary | Research boundary | Evaluation |
| [027](027-late-interaction-feasibility-boundary.md) | Late-interaction feasibility boundary | Research boundary | Retrieval |
| [028](028-raptor-feasibility-boundary.md) | RAPTOR feasibility boundary | Research boundary | Retrieval |
| [029](029-graphrag-feasibility-boundary.md) | GraphRAG feasibility boundary | Research boundary | Retrieval |
| [030](030-optional-otel-observability-boundary.md) | Optional OpenTelemetry observability boundary | Accepted | Observability |
| [031](031-failure-analysis-contract.md) | Per-case failure analysis contract | Accepted | Evaluation |
| [032](032-retriever-plugin-contract.md) | Versioned retriever plugin boundary | Accepted | Plugins |
| [033](033-dockerized-benchmark-boundary.md) | Dockerized benchmark boundary | Accepted | Evaluation |
| [034](034-contributor-taskfile-contract.md) | Portable contributor Taskfile contract | Accepted | Tooling |
| [035](035-atomic-ingestion-replacement.md) | Serialized atomic source replacement | Accepted | Ingestion |
| [036](036-retire-zero-cost-feature-flags.md) | Retire zero-cost feature flags, unify retrieval stages | Accepted | Retrieval / Configuration |
| [037](037-grouped-configuration-model.md) | Grouped configuration model behind flat public names | Accepted | Configuration |
| [038](038-application-and-mcp-boundaries.md) | Transport-agnostic application boundary and MCP adapter | Amended by [039](039-fastmcp-sdk-adoption.md) | Architecture / MCP |
| [039](039-fastmcp-sdk-adoption.md) | Adopt the FastMCP SDK for the MCP adapter | Accepted | Architecture / MCP |
| [040](040-request-scoped-collection-selection.md) | Request-scoped HTTP collection selection | Accepted | API / Retrieval |
| [041](041-claim-scope-applicability-filter.md) | Claim scope-applicability filtering | Accepted | Retrieval |

## Research spikes

ADRs 027–029 bound prototypes that live in [`research/`](../../research/) and are
deliberately **not** wired into ingestion, retrieval, or the API. Read the ADR
before assuming any of that code is reachable from the product.

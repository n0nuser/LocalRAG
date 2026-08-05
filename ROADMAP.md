# LocalRAG Roadmap

**Snapshot reviewed:** 2026-08-04
**Live status source:** [GitHub milestones](https://github.com/n0nuser/LocalRAG/milestones) and [GitHub issues](https://github.com/n0nuser/LocalRAG/issues)

This is a maintained contributor-facing summary, not a second issue tracker.
GitHub owns live issue state, milestone state, membership, and dependencies. The
six phases below are thematic groupings requested for navigation; they are not
additional milestones and do not imply that work in one phase must be completed
before another phase can start.

## How To Read This

- **Status** is the GitHub state at the snapshot date (`open` or `closed`).
- **Roadmap intent** describes whether the milestone is completed, active, or
  planned from the perspective of this summary. It never overrides GitHub.
- **Dependency** means a hard prerequisite stated by an issue. A related issue
  or thematic ordering is not a hard dependency unless the issue says so.
- **Exit outcome** is what a completed milestone should leave behind. The
  measurable signal is the reviewable evidence used to decide that outcome.
- Links use stable GitHub issue and milestone URLs. Refresh this page rather than
  editing issue state in prose.

## Phase Map

These phases group all ten actual GitHub milestones exactly once.

| Phase | Thematic purpose | GitHub milestones |
| --- | --- | --- |
| 1. Foundation and contributor workflow | Configuration, CLI, contributor tooling, extension boundaries, and deployment safety | Configuration, CLI, Developer Experience, Architecture & Extensibility, Production Readiness |
| 2. Retrieval quality | Retrieval behavior and quality improvements | Retrieval Improvements |
| 3. Evaluation contract and benchmarking | Dataset, metrics, reproducibility, comparison, reports, and benchmark environments | Evaluation |
| 4. Performance and efficiency | Runtime and evaluation efficiency | Performance |
| 5. Research feasibility | Research spikes and explicit boundaries for advanced retrieval | Advanced Research |
| 6. Project communication | Maintained documentation and published benchmark views | Documentation |

The map is not a delivery sequence. For example, evaluation contracts support
benchmark claims, while research spikes can proceed independently; the issue
dependency statements below are the authority for blocking relationships.

## Milestones

### [Configuration](https://github.com/n0nuser/LocalRAG/milestone/4)

- **GitHub status:** `open`; **roadmap intent:** completed milestone, with no
  currently open issue assigned.
- **Issues:** [#62](https://github.com/n0nuser/LocalRAG/issues/62) (closed).
- **Hard dependencies:** none remaining for the delivered YAML configuration.
- **Exit outcome:** structured YAML configuration is an explicit, documented
  input while preserving the existing environment/CLI configuration behavior.
- **Exit signal:** #62 is closed and the configuration examples, precedence
  rules, and tests are present in the repository.

### [CLI](https://github.com/n0nuser/LocalRAG/milestone/5)

- **GitHub status:** `open`; **roadmap intent:** completed milestone, with no
  currently open issue assigned.
- **Issues:** [#63](https://github.com/n0nuser/LocalRAG/issues/63) (closed).
- **Hard dependencies:** none remaining for the delivered commands.
- **Exit outcome:** contributors can run benchmark and collection inspection
  workflows through documented CLI commands.
- **Exit signal:** #63 is closed and the benchmark/inspect command contracts are
  covered by tests and CLI documentation.

### [Developer Experience](https://github.com/n0nuser/LocalRAG/milestone/6)

- **GitHub status:** `open`; **roadmap intent:** active.
- **Issues:** [#64](https://github.com/n0nuser/LocalRAG/issues/64) (closed) and
  [#130](https://github.com/n0nuser/LocalRAG/issues/130) (open).
- **Hard dependencies:** none remaining for the contributor command wrapper.
- **Exit outcome:** common install, format, lint, test, and local workflow
  commands have a portable Taskfile contract.
- **Exit signal:** #64 is closed; `task --list`, `task lint`, and `task test` are
  documented and exercised by the repository workflow.

### [Architecture & Extensibility](https://github.com/n0nuser/LocalRAG/milestone/9)

- **GitHub status:** `open`; **roadmap intent:** active.
- **Issues:** [#80](https://github.com/n0nuser/LocalRAG/issues/80) (closed),
  [#85](https://github.com/n0nuser/LocalRAG/issues/85) (closed),
  [#81](https://github.com/n0nuser/LocalRAG/issues/81),
  [#127](https://github.com/n0nuser/LocalRAG/issues/127),
  [#131](https://github.com/n0nuser/LocalRAG/issues/131), and
  [#132](https://github.com/n0nuser/LocalRAG/issues/132) (open).
- **Hard dependencies:** #81 consumes the stable embedding/provider boundary
  from #80; it must not redefine that contract. #85 is independent of #81.
- **Exit outcome:** stable, explicitly bounded extension seams exist without
  making optional integrations or untrusted plugin code part of the default
  install, with consistent provider identity, operational metrics, and safe
  persistence boundaries. The milestone remains open while assigned issues are open.
- **Exit signal:** #80 and #85 delivered provider and observability contracts;
  #81 exits when its selected retriever plugin contract, discovery, trust model,
  lifecycle, compatibility tests, and author documentation meet its issue
  acceptance criteria.

### [Retrieval Improvements](https://github.com/n0nuser/LocalRAG/milestone/1)

- **GitHub status:** `open`; **roadmap intent:** completed current scope, with no
  currently open issue assigned.
- **Issues:** [#57](https://github.com/n0nuser/LocalRAG/issues/57),
  [#58](https://github.com/n0nuser/LocalRAG/issues/58),
  [#59](https://github.com/n0nuser/LocalRAG/issues/59),
  [#60](https://github.com/n0nuser/LocalRAG/issues/60), and
  [#61](https://github.com/n0nuser/LocalRAG/issues/61) (all closed).
- **Hard dependencies:** the issue-level contracts govern implementation order;
  no assigned issue remains blocked.
- **Exit outcome:** LocalRAG has structural chunking, query expansion, context
  compression, metadata filtering, and bounded adaptive retrieval as explicit
  retrieval capabilities.
- **Exit signal:** all five assigned issues are closed and the behavior is
  documented in [retrieval design notes](docs/rag-retrieval.md) and ADRs.

### [Evaluation](https://github.com/n0nuser/LocalRAG/milestone/2)

- **GitHub status:** `open`; **roadmap intent:** completed current scope, with no
  currently open issue assigned.
- **Issues:** [#71](https://github.com/n0nuser/LocalRAG/issues/71),
  [#73](https://github.com/n0nuser/LocalRAG/issues/73),
  [#74](https://github.com/n0nuser/LocalRAG/issues/74),
  [#75](https://github.com/n0nuser/LocalRAG/issues/75),
  [#76](https://github.com/n0nuser/LocalRAG/issues/76),
  [#82](https://github.com/n0nuser/LocalRAG/issues/82),
  [#83](https://github.com/n0nuser/LocalRAG/issues/83),
  [#84](https://github.com/n0nuser/LocalRAG/issues/84),
  [#86](https://github.com/n0nuser/LocalRAG/issues/86), and
  [#87](https://github.com/n0nuser/LocalRAG/issues/87) (closed),
  [#126](https://github.com/n0nuser/LocalRAG/issues/126), and
  [#128](https://github.com/n0nuser/LocalRAG/issues/128) (open).
- **Hard dependencies:** #82 established dataset identity and registry; #83
  established reproducibility metadata; #84 owns the versioned result and
  comparison contract. #74, #73, #78, #76, #86, and #87 consume those contracts
  as stated in their issues. Benchmark claims depend on the #82/#84 evaluation
  contract, not on a single score or model run.
- **Exit outcome:** evaluation runs have versioned datasets, reproducible input
  selection and provenance, explicit metrics/comparison semantics, failure
  analysis, reports, and a containerized benchmark boundary.
- **Exit signal:** all assigned issues are closed; the contracts and outputs are
  documented in [evaluation metrics](docs/evaluation-metrics.md),
  [reproducibility](docs/reproducibility.md), and [evaluation reports](docs/evaluation-reports.md).
- **Policy:** RAGAS and live model evaluations are manual-only. CI and normal
  documentation checks do not trigger model calls; contributors may run the
  documented commands locally or by an explicitly dispatched/manual workflow.
  Offline evaluation and fixture benchmark checks remain suitable for routine
  verification.

### [Performance](https://github.com/n0nuser/LocalRAG/milestone/3)

- **GitHub status:** `open`; **roadmap intent:** active.
- **Issues:** [#77](https://github.com/n0nuser/LocalRAG/issues/77),
  [#78](https://github.com/n0nuser/LocalRAG/issues/78), and
  [#79](https://github.com/n0nuser/LocalRAG/issues/79) (closed),
  [#124](https://github.com/n0nuser/LocalRAG/issues/124), and
  [#129](https://github.com/n0nuser/LocalRAG/issues/129) (open).
- **Hard dependencies:** #77 required #80 and could consume #73's benchmark
  manifest; #78 required the evaluation contracts. Those dependencies are now
  closed.
- **Exit outcome:** embedding, evaluation, and retrieval workloads have measured
  efficiency paths rather than unbounded or anecdotal optimization claims.
- **Exit signal:** all three assigned issues are closed and their benchmark
  artifacts/protocols identify corpus, provider/configuration, units, and limits.

### [Advanced Research](https://github.com/n0nuser/LocalRAG/milestone/8)

- **GitHub status:** `open`; **roadmap intent:** feasibility work completed for
  the currently assigned topics, with no currently open issue assigned.
- **Issues:** [#67](https://github.com/n0nuser/LocalRAG/issues/67),
  [#68](https://github.com/n0nuser/LocalRAG/issues/68),
  [#69](https://github.com/n0nuser/LocalRAG/issues/69),
  [#70](https://github.com/n0nuser/LocalRAG/issues/70), and
  [#72](https://github.com/n0nuser/LocalRAG/issues/72) (all closed).
- **Hard dependencies:** these were research spikes, not prerequisites for the
  default retrieval path. Their issue-level feasibility boundaries govern any
  follow-up implementation.
- **Exit outcome:** GraphRAG, RAPTOR, HyDE, late interaction, and agentic
  retrieval have documented feasibility results and explicit adoption/defer
  boundaries.
- **Exit signal:** all five issues are closed; [ADR 025](docs/adr/025-hyde-retrieval-experiment.md),
  [ADR 027](docs/adr/027-late-interaction-feasibility-boundary.md),
  [ADR 028](docs/adr/028-raptor-feasibility-boundary.md), and
  [ADR 029](docs/adr/029-graphrag-feasibility-boundary.md) record the durable
  decisions. Research code is not the default retrieval implementation unless a
  later issue explicitly promotes it.

### [Documentation](https://github.com/n0nuser/LocalRAG/milestone/7)

- **GitHub status:** `open`; **roadmap intent:** active while this roadmap issue
  is reviewed.
- **Issues:** [#65](https://github.com/n0nuser/LocalRAG/issues/65) (closed) and
  [#66](https://github.com/n0nuser/LocalRAG/issues/66) (open at snapshot; this
  document is its deliverable).
- **Hard dependencies:** #66 uses current milestone and issue data; it does not
  block application behavior.
- **Exit outcome:** contributors can find the published benchmark view and a
  single, synchronized project roadmap without duplicating live tracker state.
- **Exit signal:** #65 is closed and #66 closes with this document, README link,
  refresh procedure, and reference validation. The milestone is then available
  for future documentation work rather than being declared complete by prose.

### [Production Readiness](https://github.com/n0nuser/LocalRAG/milestone/10)

- **GitHub status:** `open`; **roadmap intent:** planned.
- **Issues:** [#125](https://github.com/n0nuser/LocalRAG/issues/125),
  [#133](https://github.com/n0nuser/LocalRAG/issues/133), and
  [#134](https://github.com/n0nuser/LocalRAG/issues/134) (open).
- **Hard dependencies:** deployment persistence and readiness semantics should be
  settled before enabling multi-replica scaling; security defaults and data
  retention can proceed independently.
- **Exit outcome:** local and containerized deployments have explicit trust,
  persistence, readiness, identity, and data-lifecycle contracts suitable for
  supported production-like operation.
- **Exit signal:** all three issues close with deployment validation, secure
  defaults, and documented retention behavior.

## Current Open Work

At the snapshot date, milestone-assigned open work includes
[#81](https://github.com/n0nuser/LocalRAG/issues/81),
[#124](https://github.com/n0nuser/LocalRAG/issues/124),
[#125](https://github.com/n0nuser/LocalRAG/issues/125),
[#126](https://github.com/n0nuser/LocalRAG/issues/126),
[#127](https://github.com/n0nuser/LocalRAG/issues/127),
[#128](https://github.com/n0nuser/LocalRAG/issues/128),
[#129](https://github.com/n0nuser/LocalRAG/issues/129),
[#130](https://github.com/n0nuser/LocalRAG/issues/130),
[#131](https://github.com/n0nuser/LocalRAG/issues/131),
[#132](https://github.com/n0nuser/LocalRAG/issues/132),
[#133](https://github.com/n0nuser/LocalRAG/issues/133), and
[#134](https://github.com/n0nuser/LocalRAG/issues/134). Unmilestoned issue
[#48](https://github.com/n0nuser/LocalRAG/issues/48) tracks a RAGAS
`answer_relevancy` regression and is diagnostic work, not a roadmap milestone
claim. The roadmap publication issue is [#66](https://github.com/n0nuser/LocalRAG/issues/66)
and is expected to close when this change merges; it is shown above only to make
the snapshot auditable.

Do not treat the completed issues listed above as pending work. New work should
be represented by a GitHub issue and assigned to the appropriate milestone
before it is added here.

## Synchronization And Validation

GitHub issues and milestones are authoritative for live status, membership,
dependencies, and scope. This file is a reviewed summary. Refresh it whenever a
milestone title/state or issue membership/dependency changes, and at least once
per release or roadmap pull request. Update the snapshot date and re-check every
linked reference during that refresh.

Run the lightweight validator from the repository root:

```bash
uv run python scripts/validate_roadmap.py
```

It checks that every live milestone link occurs exactly once, titles and milestone
numbers match live GitHub data, referenced issue links exist, and each issue is
listed under its actual milestone. For offline review, pass a JSON fixture with
the same `milestones` and `issues` arrays:

```bash
uv run python scripts/validate_roadmap.py --fixture path/to/github-roadmap-fixture.json
```

The validator is a reference check, not a replacement for reviewing issue
content or rendered Markdown. It deliberately does not run RAGAS, live model
benchmarks, or create/update GitHub objects.

## Contributor Guidance

1. Start with the relevant GitHub milestone and issue. Prefer an unblocked issue
   with clear acceptance criteria; do not infer priority from the phase order.
2. Check the issue's hard dependencies and update the issue links/status in the
   roadmap only when a roadmap refresh is part of the change.
3. Keep changes small and land them through a short-lived branch and PR to
   `main`, following [contributor workflow](CONTRIBUTING.md).
4. Add or update an ADR in [`docs/adr/`](docs/adr/) for a durable architectural
   decision: a public contract, persistence/schema boundary, compatibility or
   migration policy, extension/trust boundary, or a default that changes system
   behavior. Experiments that do not establish a durable decision do not need an
   ADR, but their feasibility boundary should be documented.
5. When a decision changes package boundaries or entry points, update
   [`docs/agent-navigation.md`](docs/agent-navigation.md) in the same change.

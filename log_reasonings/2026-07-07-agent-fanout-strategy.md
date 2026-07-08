# 2026-07-07 — Fanning out the production-RAG-hardening plan across agents

## Context

`docs/superpowers/plans/2026-07-07-production-rag-hardening.md` defines 15
tasks. User asked to fan out one agent per task. Inspection of the plan's
per-task `**Files:**` lists showed heavy overlap: nearly every task touches
`localrag/settings.py` and `.env.example`; `localrag/api/service.py` is
touched by 8 of 15 tasks; `localrag/rag/retriever.py` and
`localrag/rag/engine.py` are each touched by 4-5 tasks. True 15-way parallel
execution against the same branch would guarantee merge conflicts.

Explicit task dependencies also exist: Task 5 needs Task 2's metadata fields
(soft dependency — fields already existed conceptually, but sequencing is
safer); Task 6 needs Task 4's and Task 5's retriever shape; Task 15 needs
Task 4's `metadata_filter` explicitly; Task 14 needs Task 13's
`ResilientProvider` explicitly.

## Decision

Asked the user directly (via AskUserQuestion) rather than guessing: chose
**batched worktrees with sequential merge** — group tasks with zero file
overlap to run truly in parallel inside isolated `git worktree`s, merge each
completed task into `main` (rebasing onto `main` first if it moved since the
worktree branched), then launch the next batch. This trades some parallelism
for safety, appropriate given the file-contention density measured above.

## Consequences

- Every worktree branch needs a rebase-onto-`main` check before merge, since
  `main` moves between when a worktree is created and when its agent
  finishes — plain `git merge --ff-only` fails whenever another task landed
  in between (happened for Task 3 and Task 4 in this run).
- Doc files (`docs/rag-retrieval.md` etc.) that multiple tasks append
  sections to will conflict at rebase time even when the code changes don't
  — these are trivial "keep both sections" resolutions, not real conflicts,
  but still require a manual look rather than blind `--theirs`/`--ours`.
- A subagent hit the session's rate limit mid-task three times in this run
  (not a code bug) — see `[[2026-07-07-agent-retry-discipline]]` for the
  resulting persistence rule.

## Related

`[[2026-07-07-ragas-offline-judge]]`, `[[2026-07-07-agent-retry-discipline]]`.

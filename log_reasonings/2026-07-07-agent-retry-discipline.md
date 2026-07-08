# 2026-07-07 — Stop on repeated failure, don't grind

## Context

During the plan fan-out (see `[[2026-07-07-agent-fanout-strategy]]`), three
concurrent subagents failed mid-task with "You've hit your session limit" —
an infrastructure/rate-limit error, not a real code failure. Before that was
understood, agent transcripts showed repeated retry attempts against the same
failure. User's instruction: "if a test is failing repeatedly, be critical
instead of keep going on and on. Ask me when in doubts via a form."

## Decision

Adopted as a standing rule (saved to persistent memory, not just this log):
after 2-3 failed attempts at the same fix, stop and surface the failure
(to the user, or in a subagent's case, report back to the orchestrator)
rather than trying further variations blind. Distinguish:

- **Real logic bugs** — worth some hypothesis-driven retry, per
  `superpowers:systematic-debugging`.
- **Environment/infra errors** (session limits, network, unavailable
  service) — report immediately, do not retry-loop at all; these will not
  self-resolve by trying a different code fix.

Applied concretely later the same session: when verifying the ragas judge
fix, `.env` in the worktree had no `OPENAI_API_KEY` (by design — `.env` is
gitignored, worktrees don't inherit it); rather than declaring the fix
"probably fine" after one failed local verification, copied the real key
over, re-checked, discovered the key itself was blank in the source `.env`
too, and escalated the real ambiguity (can this be verified locally at all?)
to the user via AskUserQuestion instead of asserting success on partial
evidence.

## Consequences

- Subagent prompts for this plan's remaining tasks now explicitly instruct:
  "If you hit the same failure a third time with the same kind of fix,
  STOP and report back the exact error instead of trying further
  variations."
- Slower in the moment (asks more questions) but avoids burning tool-call
  budget/context on unwinnable retries, and avoids false "verified working"
  claims when verification was actually only partial.

## Related

`[[2026-07-07-agent-fanout-strategy]]`, `[[2026-07-07-ragas-offline-judge]]`.

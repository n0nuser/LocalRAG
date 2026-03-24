---
name: trunk-feature-workflow
description: >-
  Verifies the current branch is merged into main (when not on main), checks out
  main, pulls, optionally restores a stash after pull, then creates a new feat/*
  branch. Use when starting a new feature from trunk or the user mentions this
  git flow.
---

# Trunk: new feature branch from `main`

Hermit uses **trunk-based development**; policy lives in [`.github/CONTRIBUTING.md`](../../../.github/CONTRIBUTING.md). This skill is the **mechanical sequence** to leave a finished branch and start a new one from up-to-date `main`.

## Steps (repo root; one command per line on PowerShell)

### 1. Fetch and merged check (skip check if you are already on `main`)

```bash
git fetch origin
```

If the current branch is **not** `main`, confirm it is **already merged** into `origin/main` before you abandon it:

```bash
git merge-base --is-ancestor HEAD origin/main
```

- Exit code **0**: current `HEAD` is contained in `origin/main` → safe to move on.
- Exit code **non-zero**: not an ancestor. **Stop** unless you know the work landed on `main` another way (e.g. **squash merge** breaks this check—verify the PR is merged on GitHub, then proceed or delete the old branch after confirming).

Optional: list local branches merged into `origin/main`:

```bash
git branch --merged origin/main
```

### 2. Switch to `main` (stash only if checkout would conflict)

```bash
git checkout main
```

If Git refuses (uncommitted/untracked changes in the way), stash, then checkout again:

```bash
git stash push -u -m "wip before main"
git checkout main
```

### 3. Pull `main`, then unstash if you stashed

```bash
git pull origin main
```

If you stashed in step 2:

```bash
git stash pop
```

Resolve any conflicts, `git add` the resolutions, and continue (commit only if the stash pop produced a conflict resolution you need to record before branching—usually fix and stage, then step 4).

### 4. Create the new feature branch

Replace the name with the real topic (e.g. `feat/improve-agents-efficiency`):

```bash
git checkout -b feat/your-topic
```

Then implement, commit with [Conventional Commits](https://www.conventionalcommits.org/), push, and open a PR to `main`.

## Oops: pushed to the wrong branch (commit is fine; click was wrong)

You want the **same changes** locally and to continue from **`main`**, not to delete the work.

1. Put the last commit back into your working tree (pick one):
   - **`git reset --soft HEAD~1`** — commit becomes **staged** changes (good if you will stash and re-commit on a `feat/…` branch).
   - **`git reset --mixed HEAD~1`** — commit becomes **unstaged** changes in the working tree.

2. Stash everything (including untracked if needed):

   ```bash
   git stash push -u -m "wip moved off wrong branch"
   ```

3. If that commit had already been **pushed** to the wrong remote branch (e.g. `develop`), update the remote **while still on that branch** after the reset: your new `HEAD` is the parent commit, so:

   ```bash
   git push origin +HEAD:develop --force-with-lease
   ```

   Replace `develop` with the branch name you pushed to. Skip this step if you never pushed.

4. Continue from **§2**: `git checkout main`, **§3** `git pull origin main`, **`git stash pop`**, then **§4** `git checkout -b feat/your-topic`.

So yes: **reset → stash → (optional force-push) → `main` → pull → unstash → new feature branch.**

## Checklist

- [ ] On a non-`main` branch: merged into `origin/main` confirmed (ancestry or GitHub after squash).
- [ ] On `main` and pulled before `git checkout -b`.
- [ ] Stash popped after pull if a stash was used; conflicts resolved before real work on the new branch.

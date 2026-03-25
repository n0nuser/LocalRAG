---
name: trunk-feature-workflow
description: >-
  Use when starting a new feature (or any conventional-commit-typed branch) from
  trunk, or when the user mentions this git flow. Verifies the current branch is
  merged into main, checks out main, pulls latest, optionally restores a stash,
  then creates a new branch following Conventional Commits naming.
---

# Trunk-based development: start a new branch from `main`

This project uses **trunk-based development** with **rebase merges**. This rule
covers the mechanical sequence to safely leave a finished branch and start fresh
from an up-to-date `main`.

Branch names follow **Conventional Commits** types:
`feat/`, `fix/`, `docs/`, `chore/`, `refactor/`, `test/`, `perf/`, `ci/`

---

## Steps

### 1. Fetch remote state

```bash
git fetch origin
```

### 2. Confirm the current branch is merged (skip if already on `main`)

Because the project uses **rebase merges**, `git merge-base --is-ancestor` is
**not reliable** — rebased commits have different SHAs than those on `main`.
Instead, verify the PR is closed and merged on GitHub, then proceed.

Optionally, check whether the branch still has unmerged commits by diffing
against `origin/main` (a clean diff means all work landed):

```bash
git diff origin/main...HEAD
```

If the diff is empty, the branch content is fully reflected in `main`.

### 3. Switch to `main` (stash only if checkout would conflict)

```bash
git checkout main
```

If Git refuses due to uncommitted changes, stash first:

```bash
git stash push -u -m "wip before main"
git checkout main
```

### 4. Pull `main`

```bash
git pull origin main
```

If you stashed in step 3, pop it now:

```bash
git stash pop
```

Resolve any conflicts, stage the resolutions with `git add`, and continue.
Only commit if the conflict resolution itself needs to be recorded — typically
just stage and move on to step 5.

### 5. Create the new branch

Use the appropriate Conventional Commits type as prefix:

```bash
git checkout -b <type>/your-topic
```

Examples:

- `feat/improve-agents-efficiency`
- `fix/null-pointer-on-startup`
- `refactor/extract-auth-service`
- `chore/update-dependencies`

Then implement, commit using [Conventional Commits](https://www.conventionalcommits.org/),
push, and open a PR targeting `main`.

---

## Recovery: pushed to the wrong branch

Your work is fine — you just need to move it to a proper branch.

**1. Undo the commit locally, keeping the changes:**

```bash
git reset --soft HEAD~1   # changes become staged
# or
git reset --mixed HEAD~1  # changes become unstaged
```

**2. Stash everything:**

```bash
git stash push -u -m "wip moved off wrong branch"
```

**3. If the commit was already pushed, force-update the remote:**

```bash
git push origin +HEAD:<wrong-branch> --force-with-lease
```

Skip this step if you never pushed.

**4. Continue from step 3:** checkout `main` → pull → pop stash → create the
correct branch.

---

## Checklist

- [ ] PR confirmed merged on GitHub before abandoning the old branch.
- [ ] On `main` and fully pulled before creating the new branch.
- [ ] Stash popped and conflicts resolved before starting real work.
- [ ] New branch name uses a valid Conventional Commits prefix.

# **Contributing**

When contributing to this repository, please first discuss the change you wish to make via issue,
email, chat, or any other method with the owners of this repository before making a change.

## Trunk-based development

The project uses **trunk-based development**: one long-lived branch, `main`, always integration-ready. Work happens on **short-lived branches** (for example `feat/…`, `fix/…`) that land on `main` quickly (ideally within days), behind pull requests and green CI.

**Why no `develop`:** an extra long-lived integration branch delays integration, hides conflicts, and trains people to merge big batches. Trunk means integrate to `main` often in small slices; feature branches exist only for the lifetime of a PR (or a very short spike).

* **Branch from** `main`, **open PRs to** `main` only. Do **not** use a long-lived `develop` (or similar) for day-to-day work. If you have local `develop` left over, delete it after its work is on `main` (or rescue unmerged commits yourself). For the routine “merged branch → new `feat/…` from `main`” steps, see [AGENTS.md](../AGENTS.md) and `.cursor/skills/trunk-feature-workflow/SKILL.md`.
* Prefer **small, incremental changes** so `main` stays shippable and reviews stay small.
* **No merge commits:** do **not** integrate `main` into your branch with `git merge`. Use **`git rebase origin/main`** (after `git fetch origin`) so history stays linear. On GitHub, maintainers must use **Rebase and merge** or **Squash and merge** only—never **Create a merge commit**. Merge commits are disabled in [`settings.yml`](settings.yml) unless you change that file. If you do not use the [Probot Settings](https://github.com/probot/settings) app, mirror the same options under **Settings → General → Pull Requests** in GitHub.
* **Releases:** tag or release from `main` when you cut a version; follow [Semantic Versioning](https://semver.org/) for version numbers.

Also, the project uses pre-commit hooks to ensure the code quality. Install them with:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

The second line registers the [Conventional Commits](https://www.conventionalcommits.org/) checker so invalid commit messages are rejected before the commit is created.

## Commits

The project uses the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This is a lightweight convention on top of commit messages. It provides an easy set of rules for creating an explicit commit history. This convention dovetails with [SemVer](https://semver.org/), by describing the features, fixes, and breaking changes made in commit messages.

Each commit message should be conceptually unique and should be able to be understood by itself. It should only contain changes related to the name of the commit. If you need to make a commit with multiple changes that are independent (conceptually), you should split it into multiple commits.

## Pull request process

1. **Update `main` locally:** `git checkout main && git pull origin main`.
2. **Create a short-lived branch** from `main`, e.g. `feat/123-short-topic`, `fix/456-bug-name`, or `chore/docs-readme` (prefix is a hint, not a substitute for [Conventional Commits](#commits) on the actual commit messages).
3. Install pre-commit: `pre-commit install` and `pre-commit install --hook-type commit-msg` (see above).
4. **Rebase on `main`** before opening or updating a PR so conflicts are resolved early: `git fetch origin` then `git rebase origin/main` (never `git merge origin/main` for this).
5. Commit with conventional messages, e.g. `git commit -m 'feat: add some AmazingFeature'`.
6. Push and open a **pull request into `main`**.

Bug fixes and small hotfixes follow the same flow: branch from `main`, PR to `main`. Reserve direct pushes to `main` for maintainers and emergencies only, if your team allows them at all.

## Issue Report Process

1. Go to the project's issues.
2. Select the template that better fits your issue.
3. Read the instructions carefully and write within the template guidelines.
4. Submit it and wait for support.

## GitHub Projects

You can use GitHub Projects to manage the project's tasks.

# **Contributing**

When contributing to this repository, please first discuss the change you wish to make via issue,
email, chat, or any other method with the owners of this repository before making a change.

## Trunk-based development

The project uses **trunk-based development**: one long-lived branch, `main`, always integration-ready. Work happens on **short-lived branches** that merge back to `main` quickly (ideally within days), behind pull requests and green CI.

* **Branch from** `main`, **open PRs to** `main` only. There is no `develop` (or similar) integration branch.
* Prefer **small, incremental changes** so `main` stays shippable and reviews stay small.
* **Merge style:** maintainers may use **squash** or **rebase** when merging PRs (both keep `main` linear; rebase preserves per-commit history from the branch). Merge commits are disabled in [`settings.yml`](settings.yml) unless you change that file. If you do not use the [Probot Settings](https://github.com/probot/settings) app, turn the same options on under **Settings → General → Pull Requests** in GitHub.
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
4. **Rebase or merge `main` into your branch** before opening or updating a PR so conflicts are resolved early (`git fetch origin && git rebase origin/main` or `git merge origin/main`).
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

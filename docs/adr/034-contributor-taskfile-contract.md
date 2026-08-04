# ADR 034: Portable Contributor Taskfile Contract

## Status

Accepted

## Context

Contributor checks and local workflows currently require remembering the exact
`uv`, CLI, pytest, Ruff, mypy, and Docker Compose commands. A workflow wrapper
is useful only if it remains portable and does not become a second application
implementation. Docker also has a host-specific WSL2 override that must not be
silently applied to every developer.

## Decision

Maintain a root `Taskfile.yml` as the discoverable contributor command
contract. Tasks delegate to existing project entry points and stop on the first
failed command, preserving its nonzero exit code. Variables are overridable for
tool executables, paths, service URLs, Compose project/files, datasets, models,
collections, sample sizes, and extra arguments.

The default Docker tasks use only `docker-compose.yml`, preserve named volumes
on `docker-down`, and provide the explicit `docker-clean` operation for volume
removal. A host may opt into a configured Compose override with
`COMPOSE_OVERRIDE`; no username or absolute host path is embedded in the
Taskfile. RAGAS evaluation and live benchmark tasks remain manual operations.

## Consequences

The Taskfile gives contributors one portable entry point across supported
Linux, macOS, Windows PowerShell, and WSL2 environments, subject to the
underlying tools being installed. It adds a small maintenance contract: task
names, variable defaults, destructive behavior, and delegated commands should
change deliberately and be documented in the README and contributor guide.
Service health remains a prerequisite for integration tasks and is not hidden
inside the non-service CI smoke validation.

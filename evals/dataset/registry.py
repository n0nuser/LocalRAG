"""Registry mapping dataset IDs to manifest files on disk.

Selecting a dataset never requires editing runner code — register a new
manifest path here (or point ``EVALS_DATASET_DIR`` at a directory of them) and
it becomes selectable via ``--dataset``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from evals.dataset.errors import DatasetNotFoundError, DatasetValidationError
from evals.dataset.schema import DatasetManifest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# dataset_id -> {dataset_version -> path}. Populated by discover_fixtures() at
# import time so `--dataset <id>` needs no code change to add a fixture.
_REGISTRY: dict[str, dict[str, Path]] = {}


def _load_manifest_file(path: Path) -> DatasetManifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        message = f"{path}: could not read/parse manifest: {exc}"
        raise DatasetValidationError(message) from exc
    try:
        return DatasetManifest.model_validate(raw)
    except ValidationError as exc:
        message = f"{path}: manifest failed validation:\n{exc}"
        raise DatasetValidationError(message) from exc


def register_manifest(path: Path) -> DatasetManifest:
    """Validate and register a manifest file. Returns the loaded manifest."""
    manifest = _load_manifest_file(path)
    _REGISTRY.setdefault(manifest.dataset_id, {})[manifest.dataset_version] = path
    return manifest


def discover_fixtures(directory: Path = FIXTURES_DIR) -> None:
    """Register every ``*.json`` manifest in a directory."""
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.json")):
        register_manifest(path)


def load_dataset(dataset_id: str, version: str | None = None) -> DatasetManifest:
    """Load a registered dataset by ID, defaulting to its highest registered version.

    "Highest" is lexical-max over registered version strings — fixtures in
    this repo use plain ``MAJOR.MINOR.PATCH`` tags, which sort correctly that
    way. A registry with non-semver tags should always pass ``version``
    explicitly rather than relying on default selection.
    """
    versions = _REGISTRY.get(dataset_id)
    if not versions:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        message = f"unknown dataset_id {dataset_id!r} (available: {available})"
        raise DatasetNotFoundError(message)

    resolved_version = version or max(versions)
    path = versions.get(resolved_version)
    if path is None:
        available = ", ".join(sorted(versions)) or "(none)"
        message = (
            f"dataset {dataset_id!r} has no version {resolved_version!r} (available: {available})"
        )
        raise DatasetNotFoundError(message)
    return _load_manifest_file(path)


def registered_datasets() -> dict[str, list[str]]:
    """dataset_id -> sorted list of registered versions, for CLI help/errors."""
    return {dataset_id: sorted(versions) for dataset_id, versions in _REGISTRY.items()}


discover_fixtures()

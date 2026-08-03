"""Content checksum for a dataset manifest.

The checksum is over the manifest's own JSON, not the source file's bytes, so
formatting (key order, whitespace) never changes it — only content does.
Result files record this alongside dataset_id/version so a silent edit to a
"released" version is detectable even if the version tag wasn't bumped.
"""

from __future__ import annotations

import hashlib

from evals.dataset.schema import DatasetManifest


def manifest_checksum(manifest: DatasetManifest) -> str:
    canonical = manifest.model_dump_json(exclude={"schema_version"})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

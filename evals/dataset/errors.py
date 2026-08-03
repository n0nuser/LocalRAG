"""Errors raised while loading or selecting evaluation datasets."""

from __future__ import annotations


class DatasetError(Exception):
    """Base class for dataset registry/loader failures."""


class DatasetValidationError(DatasetError):
    """A manifest file failed schema or cross-reference validation."""


class DatasetNotFoundError(DatasetError):
    """The requested dataset_id/version is not registered."""


class OfflineArtifactsMissingError(DatasetError):
    """Offline mode requires stored answer/context artifacts a record doesn't have."""

    def __init__(self, record_id: str, field: str) -> None:
        self.record_id = record_id
        self.field = field
        super().__init__(
            f"record {record_id!r} has no offline {field} — cannot run in --offline mode"
        )

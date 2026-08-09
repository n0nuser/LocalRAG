"""Versioned dataset manifest and record schema for evaluation datasets.

A manifest describes *what* a dataset is (identity, provenance, license) and
*how* its records are organized (splits). Records carry the question, the
reference answer, relevance judgments over context, and optional pre-built
offline artifacts. This schema is the contract #83 (reproducibility metadata),
#74 (metrics), #86 (failure analysis), and #87 (containerized benchmarks) all
build on — changing field meaning here is a breaking change for all of them.

See docs/eval-datasets.md for authoring guidance and the registry contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evals.dataset.errors import DatasetNotFoundError

# Bump when a change to this schema is not purely additive (i.e. a loader
# written against an older SCHEMA_VERSION could silently misinterpret data).
SCHEMA_VERSION = 1


class Citation(BaseModel):
    """A source reference a record's relevance judgments point at.

    ``citation_id`` is scoped to the manifest that declares it — judgments
    reference citations by this ID, and the loader rejects dangling references.
    """

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source: str = Field(description="Human-readable source identifier, e.g. a doc title or URI.")
    text: str = Field(description="The cited passage, verbatim.")


class RelevanceJudgment(BaseModel):
    """Ground-truth relevance of one citation to one record.

    ``graded`` scores use the manifest's declared scale (see
    ``DatasetManifest.judgment_scale``); ``binary`` scores are 0 or 1.
    """

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    relevant: bool
    grade: float | None = Field(
        default=None,
        description="Graded relevance score. Required when the split's judgment_type is 'graded'.",
    )


class DatasetRecord(BaseModel):
    """One evaluation example.

    ``record_id`` is stable across dataset versions that keep the same
    question — result files and #82 consumers key off this ID, not list
    position, so record order in the source file is never semantically
    meaningful.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    question: str
    reference_answer: str = Field(description="Ground-truth answer used as the RAGAS reference.")
    reference_answers: list[str] | None = Field(
        default=None,
        description="Optional multi-reference answers; metrics use the best reference score.",
    )
    citations: list[Citation] = Field(default_factory=list)
    judgments: list[RelevanceJudgment] = Field(default_factory=list)
    answer_citation_ids: list[str] | None = Field(
        default=None,
        description=(
            "Stable citation IDs expected in the evaluated answer; absent means unavailable."
        ),
    )

    # Optional pre-built artifacts for offline mode. If absent, offline mode
    # falls back to reference_answer as the answer and cited texts as context.
    offline_answer: str | None = None
    offline_contexts: list[str] | None = None
    offline_retrieved_citation_ids: list[str] | None = Field(
        default=None,
        description=(
            "Citation IDs offline mode should treat as retrieved. Absent means every "
            "declared citation, which asserts perfect retrieval — state this explicitly "
            "for records whose point is that retrieval missed something."
        ),
    )

    @field_validator("record_id")
    @classmethod
    def _record_id_not_blank(cls, value: str) -> str:
        if not value.strip():
            message = "record_id must not be blank"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def _judgments_reference_known_citations(self) -> DatasetRecord:
        known_ids = {c.citation_id for c in self.citations}
        for judgment in self.judgments:
            if judgment.citation_id not in known_ids:
                message = (
                    f"record {self.record_id!r}: judgment references unknown "
                    f"citation_id {judgment.citation_id!r}"
                )
                raise ValueError(message)
        for citation_id in self.offline_retrieved_citation_ids or []:
            if citation_id not in known_ids:
                message = (
                    f"record {self.record_id!r}: offline_retrieved_citation_ids references "
                    f"unknown citation_id {citation_id!r}"
                )
                raise ValueError(message)
        return self

    def offline_context_texts(self) -> list[str]:
        """Contexts to use in offline mode: explicit override, else cited texts."""
        if self.offline_contexts is not None:
            return self.offline_contexts
        return [c.text for c in self.citations]

    def reference_answers_or_default(self) -> list[str]:
        """Return declared references without changing the legacy field contract."""
        return self.reference_answers or [self.reference_answer]

    def relevant_citation_ids(self) -> list[str]:
        """Return citation IDs marked relevant by the #82 annotation contract."""
        return [judgment.citation_id for judgment in self.judgments if judgment.relevant]

    def citation_texts(self) -> dict[str, str]:
        """Map declared citation IDs to their passages, for the retrieval-recall join."""
        return {citation.citation_id: citation.text for citation in self.citations}

    def offline_retrieved_ids(self) -> list[str]:
        """IDs offline mode reports as retrieved: explicit override, else every citation."""
        if self.offline_retrieved_citation_ids is not None:
            return self.offline_retrieved_citation_ids
        return [citation.citation_id for citation in self.citations]


class DatasetSplit(BaseModel):
    """A named subset of a dataset version's records (e.g. 'default', 'smoke')."""

    model_config = ConfigDict(extra="forbid")

    name: str
    record_ids: list[str] = Field(
        description="record_id values from this manifest's records, defining membership and order."
    )


class DatasetManifest(BaseModel):
    """Identity, provenance, and content of one dataset version.

    ``dataset_id`` is stable across versions of the same logical dataset;
    ``dataset_version`` is an immutable snapshot — editing records without
    bumping the version breaks any result file that recorded the old checksum.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    dataset_id: str
    dataset_version: str = Field(description="Immutable version tag, e.g. '1.0.0'.")
    description: str
    source: str = Field(description="Where these records came from (authored, scraped, etc).")
    license: str
    judgment_type: str = Field(
        default="binary", description="'binary' or 'graded' — see RelevanceJudgment.grade."
    )
    records: list[DatasetRecord]
    splits: list[DatasetSplit]

    @field_validator("judgment_type")
    @classmethod
    def _judgment_type_known(cls, value: str) -> str:
        if value not in {"binary", "graded"}:
            message = f"judgment_type must be 'binary' or 'graded', got {value!r}"
            raise ValueError(message)
        return value

    @model_validator(mode="after")
    def _validate_cross_references(self) -> DatasetManifest:
        self._check_unique_record_ids()
        self._check_graded_judgments_have_grades()
        self._check_splits()
        self._check_schema_version()
        return self

    def _check_unique_record_ids(self) -> None:
        seen: set[str] = set()
        for record in self.records:
            if record.record_id in seen:
                message = f"duplicate record_id: {record.record_id!r}"
                raise ValueError(message)
            seen.add(record.record_id)

    def _check_graded_judgments_have_grades(self) -> None:
        if self.judgment_type != "graded":
            return
        for record in self.records:
            for judgment in record.judgments:
                if judgment.grade is None:
                    message = (
                        f"record {record.record_id!r}: judgment for "
                        f"{judgment.citation_id!r} missing grade (judgment_type=graded)"
                    )
                    raise ValueError(message)

    def _check_splits(self) -> None:
        known_record_ids = {r.record_id for r in self.records}
        seen_names: set[str] = set()
        for split in self.splits:
            if split.name in seen_names:
                message = f"duplicate split name: {split.name!r}"
                raise ValueError(message)
            seen_names.add(split.name)
            for record_id in split.record_ids:
                if record_id not in known_record_ids:
                    message = f"split {split.name!r} references unknown record_id {record_id!r}"
                    raise ValueError(message)

    def _check_schema_version(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            message = (
                f"unsupported schema_version {self.schema_version} "
                f"(this loader supports {SCHEMA_VERSION})"
            )
            raise ValueError(message)

    def split(self, name: str) -> list[DatasetRecord]:
        """Records belonging to a named split, in the split's declared order."""
        matches = [s for s in self.splits if s.name == name]
        if not matches:
            available = ", ".join(s.name for s in self.splits) or "(none)"
            message = (
                f"unknown split {name!r} for dataset {self.dataset_id!r} (available: {available})"
            )
            raise DatasetNotFoundError(message)
        by_id = {r.record_id: r for r in self.records}
        return [by_id[record_id] for record_id in matches[0].record_ids]

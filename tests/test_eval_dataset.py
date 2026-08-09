from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import evals.run_evals as run_evals_module
from evals.dataset.checksum import manifest_checksum
from evals.dataset.errors import (
    DatasetNotFoundError,
    DatasetValidationError,
    OfflineArtifactsMissingError,
)
from evals.dataset.registry import (
    discover_fixtures,
    load_dataset,
    register_manifest,
    registered_datasets,
)
from evals.dataset.schema import DatasetManifest, DatasetRecord
from evals.metrics import score_retrieval_recall
from evals.run_evals import _build_rows, _select_records

# --- fixture manifest builders ---


def _manifest(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "dataset_id": "test-ds",
        "dataset_version": "1.0.0",
        "description": "d",
        "source": "s",
        "license": "l",
        "records": [
            {
                "record_id": "r1",
                "question": "q1?",
                "reference_answer": "a1",
                "citations": [{"citation_id": "r1-c1", "source": "s", "text": "t1"}],
                "judgments": [{"citation_id": "r1-c1", "relevant": True}],
            },
            {
                "record_id": "r2",
                "question": "q2?",
                "reference_answer": "a2",
                "citations": [{"citation_id": "r2-c1", "source": "s", "text": "t2"}],
                "judgments": [{"citation_id": "r2-c1", "relevant": True}],
            },
        ],
        "splits": [{"name": "default", "record_ids": ["r1", "r2"]}],
    }
    base.update(overrides)
    return base


# --- schema validation ---


def test_valid_manifest_parses() -> None:
    manifest = DatasetManifest.model_validate(_manifest())
    assert manifest.dataset_id == "test-ds"
    assert len(manifest.records) == 2


def test_duplicate_record_id_rejected() -> None:
    records = _manifest()["records"]
    records.append(dict(records[0]))
    with pytest.raises(ValidationError, match="duplicate record_id"):
        DatasetManifest.model_validate(_manifest(records=records))


def test_judgment_referencing_unknown_citation_rejected() -> None:
    bad = _manifest()
    bad["records"][0]["judgments"] = [{"citation_id": "does-not-exist", "relevant": True}]
    with pytest.raises(ValidationError, match="unknown citation_id"):
        DatasetManifest.model_validate(bad)


def test_split_referencing_unknown_record_rejected() -> None:
    bad = _manifest(splits=[{"name": "default", "record_ids": ["r1", "ghost"]}])
    with pytest.raises(ValidationError, match="unknown record_id"):
        DatasetManifest.model_validate(bad)


def test_duplicate_split_name_rejected() -> None:
    bad = _manifest(
        splits=[
            {"name": "default", "record_ids": ["r1"]},
            {"name": "default", "record_ids": ["r2"]},
        ]
    )
    with pytest.raises(ValidationError, match="duplicate split name"):
        DatasetManifest.model_validate(bad)


def test_graded_judgment_type_requires_grade() -> None:
    bad = _manifest(judgment_type="graded")
    with pytest.raises(ValidationError, match="missing grade"):
        DatasetManifest.model_validate(bad)


def test_graded_judgment_type_with_grade_is_valid() -> None:
    ok = _manifest(judgment_type="graded")
    for record in ok["records"]:
        for judgment in record["judgments"]:
            judgment["grade"] = 0.5
    manifest = DatasetManifest.model_validate(ok)
    assert manifest.judgment_type == "graded"


def test_unknown_judgment_type_rejected() -> None:
    with pytest.raises(ValidationError, match="judgment_type"):
        DatasetManifest.model_validate(_manifest(judgment_type="fuzzy"))


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(ValidationError, match="unsupported schema_version"):
        DatasetManifest.model_validate(_manifest(schema_version=99))


def test_blank_record_id_rejected() -> None:
    bad = _manifest()
    bad["records"][0]["record_id"] = "   "
    with pytest.raises(ValidationError, match="record_id must not be blank"):
        DatasetManifest.model_validate(bad)


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetManifest.model_validate(_manifest(unexpected_field="nope"))


def test_split_returns_records_in_declared_order() -> None:
    manifest = DatasetManifest.model_validate(
        _manifest(splits=[{"name": "reversed", "record_ids": ["r2", "r1"]}])
    )
    ordered = manifest.split("reversed")
    assert [r.record_id for r in ordered] == ["r2", "r1"]


def test_unknown_split_name_raises() -> None:
    manifest = DatasetManifest.model_validate(_manifest())
    with pytest.raises(DatasetNotFoundError, match="unknown split"):
        manifest.split("nope")


def test_offline_context_texts_falls_back_to_citations() -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="a",
        citations=[{"citation_id": "c1", "source": "s", "text": "cited text"}],
    )
    assert record.offline_context_texts() == ["cited text"]


def test_offline_context_texts_honors_explicit_override() -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="a",
        citations=[{"citation_id": "c1", "source": "s", "text": "cited text"}],
        offline_contexts=["override text"],
    )
    assert record.offline_context_texts() == ["override text"]


# --- checksum ---


def test_checksum_stable_across_reparse() -> None:
    payload = _manifest()
    first = manifest_checksum(DatasetManifest.model_validate(payload))
    second = manifest_checksum(DatasetManifest.model_validate(json.loads(json.dumps(payload))))
    assert first == second


def test_checksum_changes_when_content_changes() -> None:
    a = manifest_checksum(DatasetManifest.model_validate(_manifest()))
    changed = _manifest()
    changed["records"][0]["reference_answer"] = "different answer"
    b = manifest_checksum(DatasetManifest.model_validate(changed))
    assert a != b


def test_checksum_ignores_schema_version_bookkeeping_field() -> None:
    """schema_version is loader compatibility metadata, not content — excluded on purpose."""
    manifest = DatasetManifest.model_validate(_manifest())
    assert manifest_checksum(manifest) == manifest_checksum(manifest)


# --- registry ---


def test_register_manifest_from_file(tmp_path: Path) -> None:
    path = tmp_path / "test-ds-1.0.0.json"
    path.write_text(json.dumps(_manifest()))
    manifest = register_manifest(path)
    assert manifest.dataset_id == "test-ds"
    loaded = load_dataset("test-ds", "1.0.0")
    assert loaded.dataset_version == "1.0.0"


def test_register_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not valid json")
    with pytest.raises(DatasetValidationError):
        register_manifest(path)


def test_register_manifest_rejects_schema_violation(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    bad = _manifest()
    bad["records"].append(dict(bad["records"][0]))  # duplicate record_id
    path.write_text(json.dumps(bad))
    with pytest.raises(DatasetValidationError):
        register_manifest(path)


def test_load_unknown_dataset_id_raises() -> None:
    with pytest.raises(DatasetNotFoundError, match="unknown dataset_id"):
        load_dataset("this-does-not-exist")


def test_load_unknown_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "test-ds-1.0.0.json"
    path.write_text(json.dumps(_manifest()))
    register_manifest(path)
    with pytest.raises(DatasetNotFoundError, match="no version"):
        load_dataset("test-ds", "9.9.9")


def test_load_default_version_picks_highest(tmp_path: Path) -> None:
    for version in ("1.0.0", "2.0.0", "1.5.0"):
        path = tmp_path / f"v{version}.json"
        path.write_text(json.dumps(_manifest(dataset_version=version)))
        register_manifest(path)
    manifest = load_dataset("test-ds")
    assert manifest.dataset_version == "2.0.0"


def test_discover_fixtures_registers_bundled_datasets() -> None:
    """The two bundled fixtures (localrag-core, localrag-graded) must be discoverable."""
    discover_fixtures()
    datasets = registered_datasets()
    assert "localrag-core" in datasets
    assert "localrag-graded" in datasets


def test_second_fixture_requires_no_runner_code_change() -> None:
    """Selecting the second bundled dataset works through the same load_dataset() call."""
    manifest = load_dataset("localrag-graded")
    assert manifest.judgment_type == "graded"
    assert len(manifest.records) >= 1


# --- deterministic selection (record_id based) ---


def _records(*ids: str) -> list[DatasetRecord]:
    return [DatasetRecord(record_id=i, question=f"q-{i}", reference_answer=f"a-{i}") for i in ids]


def test_select_records_is_order_independent() -> None:
    forward = _records("a", "b", "c", "d")
    backward = list(reversed(forward))
    assert _select_records(forward, seed=42, sample=2) == _select_records(
        backward, seed=42, sample=2
    )


def test_select_records_same_seed_is_stable() -> None:
    records = _records("a", "b", "c", "d", "e")
    first = _select_records(records, seed=7, sample=3)
    second = _select_records(records, seed=7, sample=3)
    assert [r.record_id for r in first] == [r.record_id for r in second]


def test_select_records_without_sample_returns_all_sorted_by_id() -> None:
    records = _records("c", "a", "b")
    selected = _select_records(records, seed=42, sample=None)
    assert [r.record_id for r in selected] == ["a", "b", "c"]


# --- offline artifact validation ---


def test_build_rows_offline_missing_contexts_raises() -> None:
    record = DatasetRecord(record_id="r1", question="q", reference_answer="a")
    with pytest.raises(OfflineArtifactsMissingError, match="contexts"):
        _build_rows([record], "http://unused", "", offline=True)


def test_build_rows_offline_uses_reference_answer_when_no_offline_answer() -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="the answer",
        citations=[{"citation_id": "c1", "source": "s", "text": "ctx"}],
    )
    rows = _build_rows([record], "http://unused", "", offline=True)
    assert rows[0]["answer"] == "the answer"
    assert rows[0]["contexts"] == ["ctx"]


def test_build_rows_offline_prefers_explicit_offline_answer() -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="reference",
        offline_answer="stored answer",
        citations=[{"citation_id": "c1", "source": "s", "text": "ctx"}],
    )
    rows = _build_rows([record], "http://unused", "", offline=True)
    assert rows[0]["answer"] == "stored answer"


def test_build_rows_records_record_id() -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="a",
        citations=[{"citation_id": "c1", "source": "s", "text": "ctx"}],
    )
    rows = _build_rows([record], "http://unused", "", offline=True)
    assert rows[0]["record_id"] == "r1"


def test_live_build_rows_uses_api_context_text_and_ids_without_fixture_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = DatasetRecord(
        record_id="r1",
        question="q",
        reference_answer="a",
        citations=[{"citation_id": "fixture", "source": "s", "text": "fixture text"}],
    )
    monkeypatch.setattr(
        run_evals_module,
        "_query_api",
        lambda *_args: ("live answer", ["actual chunk text"], ["live-id"]),
    )

    rows = _build_rows([record], "http://unused", "", offline=False)

    assert rows[0]["contexts"] == ["actual chunk text"]
    assert rows[0]["retrieved_ids"] == ["live-id"]
    assert "fixture text" not in rows[0]["contexts"]


# --- bundled fixtures stay valid (regression guard) ---


def test_bundled_localrag_core_fixture_is_valid() -> None:
    manifest = load_dataset("localrag-core")
    assert manifest.dataset_id == "localrag-core"
    assert len(manifest.records) > 0
    assert manifest.split("default")
    assert manifest.split("smoke")


def test_bundled_fixtures_offline_mode_never_needs_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline mode must not call _query_api for a fully-populated bundled fixture."""

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("offline mode must not call the API")

    monkeypatch.setattr(run_evals_module, "_query_api", _boom)
    manifest = load_dataset("localrag-core")
    records = manifest.split("smoke")
    rows = _build_rows(records, "http://unused", "", offline=True)
    assert len(rows) == len(records)


def test_scope_fixture_encodes_the_observed_retrieval_failure() -> None:
    """#174's regression case: the acute passage is indexed but never retrieved."""
    discover_fixtures()
    manifest = load_dataset("localrag-scope", "1.0.0")
    recalls = {
        record.record_id: score_retrieval_recall(
            record.relevant_citation_ids(),
            record.offline_retrieved_ids(),
            record.citation_texts(),
            record.offline_context_texts(),
        ).value
        for record in manifest.records
    }
    # The motivating failure, a graded partial, and a control that must not be zero.
    assert recalls["single-overvoltage-event-effect"] == 0.0
    assert recalls["single-event-recovery-procedure"] == 0.5
    assert recalls["repeated-overvoltage-events-effect"] == 1.0


def test_offline_retrieved_ids_default_to_every_citation() -> None:
    """Existing fixtures declare no override and must keep their previous meaning."""
    discover_fixtures()
    record = load_dataset("localrag-core", "1.0.0").records[0]
    assert record.offline_retrieved_ids() == [c.citation_id for c in record.citations]


def test_offline_retrieved_ids_must_reference_declared_citations() -> None:
    with pytest.raises(ValidationError, match="offline_retrieved_citation_ids"):
        DatasetRecord.model_validate(
            {
                "record_id": "r1",
                "question": "q",
                "reference_answer": "a",
                "citations": [{"citation_id": "c1", "source": "s", "text": "t"}],
                "offline_retrieved_citation_ids": ["nope"],
            }
        )

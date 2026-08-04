from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from evals.report import generate_report
from localrag.cli.app import app


def _matrix(path: Path, *, run_id: str, dataset_id: str = "fixture") -> Path:
    payload = {
        "schema_version": 1,
        "matrix_id": "matrix",
        "run_id": run_id,
        "profile": "fixture",
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_version": "1",
            "split": "smoke",
            "checksum": "checksum",
        },
        "corpus": {"identity": dataset_id, "checksum": "checksum"},
        "code_revision": "abc",
        "code_dirty": False,
        "model": {"provider": "ollama", "generation": "model"},
        "effective_config": {"retrieval_mode": "hybrid"},
        "mode": "offline",
        "supported_dimensions": {},
        "seed": 42,
        "status": "complete",
        "started_at": "2026-08-04T00:00:00Z",
        "finished_at": "2026-08-04T00:00:01Z",
        "cases": [
            {
                "case_id": "case-1",
                "run_id": run_id,
                "matrix_id": "matrix",
                "dataset": {
                    "dataset_id": dataset_id,
                    "dataset_version": "1",
                    "split": "smoke",
                    "checksum": "checksum",
                },
                "corpus": {"identity": dataset_id, "checksum": "checksum"},
                "code_revision": "abc",
                "code_dirty": False,
                "model": {"provider": "ollama"},
                "supported_dimensions": {},
                "seed": 42,
                "effective_config": {"retrieval_mode": "hybrid"},
                "status": "failed",
                "started_at": "2026-08-04T00:00:00Z",
                "finished_at": "2026-08-04T00:00:01Z",
                "metrics": {"quality": 0.8},
                "latency": {"value": 1.25, "unit": "seconds"},
                "resources": {"memory_mb": 12},
                "error": {"type": "RuntimeError", "message": "<script>alert(1)</script>"},
            }
        ],
        "artifact_paths": {"result": "/private/source.json"},
        "exit_code": 1,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_handles_multiple_runs_and_optional_fields(tmp_path: Path) -> None:
    first = _matrix(tmp_path / "first.json", run_id="run-1")
    second = _matrix(tmp_path / "second.json", run_id="run-2")
    output = tmp_path / "report.html"

    result = generate_report([second, first], output)

    assert result.errors == []
    assert result.run_count == 2
    html = output.read_text(encoding="utf-8")
    assert "run-1" in html
    assert "run-2" in html
    assert "quality" in html
    assert "Unavailable" in html
    assert "1.25" in html
    assert "seconds" in html
    assert "memory_mb" in html
    assert "failed" in html


def test_report_surfaces_malformed_input_and_empty_input(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    output = tmp_path / "report.html"

    result = generate_report([bad], output)
    assert result.run_count == 0
    assert "bad.json" in result.errors[0]
    assert "No benchmark runs" in output.read_text(encoding="utf-8")

    empty = generate_report([], tmp_path / "empty.html")
    assert empty.errors == []
    assert empty.run_count == 0


def test_report_is_deterministic_escapes_content_and_has_no_network(tmp_path: Path) -> None:
    source = _matrix(tmp_path / "source.json", run_id="<run>")
    output_a = tmp_path / "a.html"
    output_b = tmp_path / "b.html"

    generate_report([source], output_a)
    generate_report([source], output_b)
    html = output_a.read_text(encoding="utf-8")

    assert html == output_b.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "https://" not in html
    assert "http://" not in html
    assert "/private/source.json" not in html


def test_incompatible_runs_are_reported_without_comparison(tmp_path: Path) -> None:
    first = _matrix(tmp_path / "first.json", run_id="run-1")
    second = _matrix(tmp_path / "second.json", run_id="run-2", dataset_id="other")

    result = generate_report([first, second], tmp_path / "report.html")

    assert result.incompatible_runs
    assert "incompatible" in tmp_path.joinpath("report.html").read_text(encoding="utf-8").lower()


def test_nan_and_strict_input_errors_are_safe(tmp_path: Path) -> None:
    source = _matrix(tmp_path / "nan.json", run_id="nan")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["cases"][0]["metrics"]["quality"] = "NaN"
    source.write_text(json.dumps(payload), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")

    result = generate_report([source, bad], tmp_path / "report.html")

    assert result.run_count == 1
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "NaN" not in html
    assert "http://" not in html


def test_canonical_evaluation_result_and_cli_strict_mode(tmp_path: Path) -> None:
    result_path = tmp_path / "evaluation.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "evaluation-1",
                "timestamp": "2026-08-04T00:00:00Z",
                "dataset": {
                    "dataset_id": "fixture",
                    "dataset_version": "1",
                    "split": "smoke",
                    "checksum": "checksum",
                },
                "selected_ids": ["case-1"],
                "metrics": [
                    {
                        "descriptor": {
                            "name": "quality",
                            "direction": "higher_is_better",
                            "threshold": 0.7,
                        },
                        "value": 0.8,
                        "cases": {"case-1": 0.8},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.html"
    generated = generate_report([result_path], output)
    assert generated.run_count == 1
    assert "pass" in output.read_text(encoding="utf-8")

    bad = tmp_path / "bad.json"
    bad.write_text("bad", encoding="utf-8")
    cli_output = tmp_path / "cli.html"
    command = CliRunner().invoke(
        app, ["report", str(bad), "--strict", "--output", str(cli_output)]
    )
    assert command.exit_code == 1
    assert cli_output.exists()

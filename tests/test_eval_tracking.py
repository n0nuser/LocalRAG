from pathlib import Path

import pytest

from evals.tracking import TrackingConfig, TrackingSession, redact


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def start_parent(self, run_id: str, params: dict[str, object]) -> None:
        self.calls.append(("start_parent", (run_id, params)))

    def start_case(self, case_id: str, params: dict[str, object]) -> None:
        self.calls.append(("start_case", (case_id, params)))

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.calls.append(("metrics", metrics))

    def log_artifact(self, path: Path) -> None:
        self.calls.append(("artifact", path.name))

    def finish_case(self, status: str, error: str | None) -> None:
        self.calls.append(("finish_case", (status, error)))

    def finish_parent(self, status: str, error: str | None) -> None:
        self.calls.append(("finish_parent", (status, error)))

    def close(self) -> None:
        self.calls.append(("close", None))


def test_redact_removes_secrets_paths_and_content() -> None:
    value = redact(
        {
            "api_key": "secret",
            "endpoint": "https://user:pass@example.test/run",
            "work_path": "/home/user/private/file.json",
            "prompt": "What is private?",
            "answer": "private answer",
            "model": "gemma3:4b",
        }
    )

    assert value == {
        "api_key": "[REDACTED]",
        "endpoint": "[REDACTED]",
        "work_path": "[REDACTED]",
        "prompt": "[REDACTED]",
        "answer": "[REDACTED]",
        "model": "gemma3:4b",
    }


def test_disabled_session_does_not_load_or_call_backend() -> None:
    backend = FakeBackend()
    session = TrackingSession(TrackingConfig(enabled=False), backend=backend)

    session.start_parent("run-1", {"model": "local"})
    session.start_case("case-1", {"prompt": "secret"})
    session.log_metrics({"score": 1.0})
    session.finish_case("complete")
    session.finish_parent("complete")
    session.close()

    assert backend.calls == []


def test_nested_lifecycle_and_artifact_selection() -> None:
    backend = FakeBackend()
    session = TrackingSession(TrackingConfig(enabled=True), backend=backend)
    artifact = Path("manifest.json")

    session.start_parent("run-1", {"prompt": "secret", "model": "local"})
    session.start_case("case-1", {"case": "one"})
    session.log_metrics({"score": 0.5})
    session.log_artifacts([Path("z.txt"), artifact, Path("a.json")])
    session.finish_case("failed", "bad")
    session.finish_parent("failed", "case failed")
    session.close()

    assert [call[0] for call in backend.calls] == [
        "start_parent",
        "start_case",
        "metrics",
        "artifact",
        "finish_case",
        "finish_parent",
        "close",
    ]
    assert backend.calls[3] == ("artifact", "manifest.json")


def test_backend_failure_is_isolated_and_cleanup_runs() -> None:
    class BrokenBackend(FakeBackend):
        def start_parent(self, run_id: str, params: dict[str, object]) -> None:
            raise RuntimeError("backend offline")

    backend = BrokenBackend()
    session = TrackingSession(TrackingConfig(enabled=True), backend=backend)

    session.start_parent("run-1", {})
    session.start_case("case-1", {})
    session.finish_parent("complete")
    session.close()

    assert backend.calls == [("close", None)]


def test_mlflow_local_extra_smoke(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    session = TrackingSession(
        TrackingConfig(enabled=True, uri=f"file:{tmp_path}", experiment="smoke"),
    )
    session.start_parent("smoke-parent", {"model": "fixture"})
    session.start_case("smoke-case", {"case": "fixture"})
    session.log_metrics({"score": 1.0})
    session.finish_case("complete")
    session.finish_parent("complete")
    session.close()

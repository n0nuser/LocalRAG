"""Optional, privacy-preserving experiment tracking for evaluation runs."""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)
_SECRET_KEYS = re.compile(
    r"(?:api[_-]?key|token|password|secret|credential|authorization)", re.IGNORECASE
)
_CONTENT_KEYS = re.compile(
    r"(?:prompt|question|answer|context|document|source|completion|response)", re.IGNORECASE
)
_PATH_KEYS = re.compile(r"(?:path|file|directory|url|endpoint|work|artifact)", re.IGNORECASE)
_CREDENTIAL_URL = re.compile(
    r"^[a-z][a-z0-9+.-]*://[^/@:]+(?::[^/@]*)?@", re.IGNORECASE
)


class TrackingBackend(Protocol):
    def start_parent(self, run_id: str, params: dict[str, object]) -> None: ...

    def start_case(self, case_id: str, params: dict[str, object]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def finish_case(self, status: str, error: str | None) -> None: ...

    def finish_parent(self, status: str, error: str | None) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    uri: str = "file:./evals/tracking"
    experiment: str = "localrag-evaluations"
    capture_content: bool = False
    retries: int = 1

    @classmethod
    def from_env(cls) -> TrackingConfig:
        return cls(
            enabled=os.getenv("EVAL_TRACKING_ENABLED", "false").lower() in {"1", "true", "yes"},
            uri=os.getenv("EVAL_TRACKING_URI", cls.uri),
            experiment=os.getenv("EVAL_TRACKING_EXPERIMENT", cls.experiment),
            capture_content=os.getenv("EVAL_TRACKING_CAPTURE_CONTENT", "false").lower()
            in {"1", "true", "yes"},
            retries=max(0, int(os.getenv("EVAL_TRACKING_RETRIES", "1"))),
        )


def redact(value: Any, *, capture_content: bool = False) -> Any:
    """Return a JSON-like value with secrets and user content removed."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            name = str(key)
            result[name] = "[REDACTED]" if (
                _SECRET_KEYS.search(name)
                or _PATH_KEYS.search(name)
                or (_CONTENT_KEYS.search(name) and not capture_content)
            ) else redact(value[key], capture_content=capture_content)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item, capture_content=capture_content) for item in value]
    if isinstance(value, str):
        if _CREDENTIAL_URL.match(value):
            return "[REDACTED]"
        return re.sub(
            r"(?i)(api[_-]?key|token|password|secret)=\S+",
            r"\1=[REDACTED]",
            value,
        )
    return value


def _backend(config: TrackingConfig) -> TrackingBackend:
    try:
        import mlflow  # type: ignore[import-not-found, PLC0415]
    except ImportError as exc:
        raise RuntimeError("MLflow tracking requires the optional 'tracking' extra") from exc
    return _MlflowBackend(mlflow, config)


class _MlflowBackend:
    def __init__(self, mlflow: Any, config: TrackingConfig) -> None:
        self._mlflow = mlflow
        mlflow.set_tracking_uri(config.uri)
        mlflow.set_experiment(config.experiment)
        self._active = False

    def start_parent(self, run_id: str, params: dict[str, object]) -> None:
        self._mlflow.start_run(run_id=run_id, run_name=run_id)
        self._active = True
        self._mlflow.log_params(params)

    def start_case(self, case_id: str, params: dict[str, object]) -> None:
        self._mlflow.start_run(run_id=case_id, run_name=case_id, nested=True)
        self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def finish_case(self, status: str, error: str | None) -> None:
        self._mlflow.set_tag("status", status)
        if error:
            self._mlflow.set_tag("error", error[:500])
        self._mlflow.end_run(status="FAILED" if status == "failed" else "FINISHED")

    def finish_parent(self, status: str, error: str | None) -> None:
        self._mlflow.set_tag("status", status)
        if error:
            self._mlflow.set_tag("error", error[:500])
        self._mlflow.end_run(status="FAILED" if status == "failed" else "FINISHED")

    def close(self) -> None:
        if self._active and self._mlflow.active_run() is not None:
            self._mlflow.end_run(status="FAILED")
        self._active = False


class TrackingSession:
    """Best-effort lifecycle boundary; tracking can never change evaluation results."""

    def __init__(
        self, config: TrackingConfig | None = None, *, backend: TrackingBackend | None = None
    ) -> None:
        self.config = config or TrackingConfig.from_env()
        self._backend = backend
        self._available = self.config.enabled

    def _call(self, method: str, *args: Any) -> None:
        if not self._available:
            return
        if self._backend is None:
            try:
                self._backend = _backend(self.config)
            except Exception as exc:  # tracking is optional infrastructure
                self._disable(exc)
                return
        for attempt in range(self.config.retries + 1):
            try:
                getattr(self._backend, method)(*args)
            except Exception as exc:  # tracking outage must not abort evaluation
                if attempt == self.config.retries:
                    self._disable(exc)
            else:
                return

    def _params(self, params: dict[str, Any]) -> dict[str, object]:
        safe = redact(params, capture_content=self.config.capture_content)
        return {key: str(value) for key, value in safe.items()}

    def _disable(self, exc: BaseException) -> None:
        LOGGER.warning("evaluation tracking disabled after backend failure: %s", redact(str(exc)))
        self._available = False

    def start_parent(self, run_id: str, params: dict[str, Any]) -> None:
        self._call("start_parent", run_id, self._params(params))

    def start_case(self, case_id: str, params: dict[str, Any]) -> None:
        self._call("start_case", case_id, self._params(params))

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._call(
            "log_metrics",
            {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            },
        )

    def log_artifacts(self, paths: list[Path]) -> None:
        # Only canonical JSON contracts are safe deterministic artifacts by default.
        for path in sorted(paths, key=lambda item: str(item)):
            if path.suffix == ".json" and path.name in {"manifest.json", "result.json"}:
                self._call("log_artifact", path)

    def finish_case(self, status: str, error: str | None = None) -> None:
        self._call("finish_case", status, redact(error) if error else None)

    def finish_parent(self, status: str, error: str | None = None) -> None:
        self._call("finish_parent", status, redact(error) if error else None)

    def close(self) -> None:
        if not self.config.enabled or self._backend is None:
            return
        try:
            self._backend.close()
        except Exception as exc:  # cleanup cannot affect evaluation
            LOGGER.warning("evaluation tracking cleanup failed: %s", redact(str(exc)))

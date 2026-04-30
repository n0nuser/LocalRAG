from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTH_URL = "http://localhost:8000/health"
HEALTH_TIMEOUT_SECONDS = 60


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_stdout(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _write_stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _emit_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        _write_stdout(result.stdout.strip())
    if result.stderr:
        _write_stderr(result.stderr.strip())


def _print_command_failure(result: subprocess.CompletedProcess[str], *, command_name: str) -> None:
    _emit_process_output(result)
    _write_stderr(f"{command_name} failed with exit code {result.returncode}.")


def _docker_available() -> bool:
    try:
        result = _run(["docker", "info"])
    except FileNotFoundError:
        _write_stdout("Docker is not installed or not on PATH. Skipping integration tests.")
        return False
    if result.returncode != 0:
        _write_stdout("Docker daemon is not reachable. Skipping integration tests.")
        return False
    return True


def _service_running(service_name: str) -> bool:
    result = _run(["docker", "compose", "ps", "--filter", "status=running", "--services"])
    if result.returncode != 0:
        _print_command_failure(result, command_name="docker compose ps")
        return False
    running_services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return service_name in running_services


def _wait_for_health() -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response = httpx.get(HEALTH_URL, timeout=3.0)
            if response.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def _api_reachable() -> bool:
    try:
        response = httpx.get(HEALTH_URL, timeout=3.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _ensure_stack_running() -> tuple[bool, int]:
    api_was_running = _service_running("localrag-api")
    if api_was_running:
        return False, 0

    started_stack = False
    up_result = _run(["docker", "compose", "up", "-d"])
    if up_result.returncode != 0:
        if _api_reachable():
            _write_stdout(
                "docker compose up -d failed, but API is already reachable on localhost:8000. "
                "Reusing running stack."
            )
        else:
            _print_command_failure(up_result, command_name="docker compose up -d")
            return False, up_result.returncode
    else:
        started_stack = True

    if _wait_for_health():
        return started_stack, 0

    _write_stderr("LocalRAG API did not become healthy within timeout.")
    if started_stack:
        down_result = _run(["docker", "compose", "down"])
        if down_result.returncode != 0:
            _print_command_failure(down_result, command_name="docker compose down")
    return started_stack, 1


def main() -> int:
    if not _docker_available():
        return 0

    started_stack, stack_status = _ensure_stack_running()
    if stack_status != 0:
        return stack_status

    test_result = _run(["uv", "run", "pytest", "-m", "integration", "-q", "--no-header"])

    if started_stack:
        down_result = _run(["docker", "compose", "down"])
        if down_result.returncode != 0:
            _print_command_failure(down_result, command_name="docker compose down")

    _emit_process_output(test_result)
    return test_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

"""Guard the wheel's import surface.

`localrag.cli` imports `evals`, so anything the CLI touches must ship in the
wheel. This regressed once: `evals/` was omitted from the distribution and every
`localrag` invocation failed with ``ModuleNotFoundError: No module named
'evals'``. The rest of the suite cannot catch it because it runs from the repo
root, where `evals/` is importable whether or not it is packaged.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]


def top_level_imports(package: Path) -> set[str]:
    """Return the top-level packages imported by every module under ``package``."""
    found: set[str] = set()
    for module in package.rglob("*.py"):
        for line in module.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("from "):
                found.add(stripped.split()[1].split(".")[0])
            elif stripped.startswith("import "):
                found.add(stripped.split()[1].split(".")[0].rstrip(","))
    return found


def test_packages_cover_every_first_party_import_of_the_shipped_cli() -> None:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    packaged = set(config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])

    imported = top_level_imports(REPO_ROOT / "localrag")
    first_party = {
        name
        for name in imported
        if (REPO_ROOT / name).is_dir() and (REPO_ROOT / name / "__init__.py").exists()
    }

    missing = first_party - packaged
    assert not missing, (
        f"{sorted(missing)} are imported by localrag/ but absent from the wheel packages; "
        "the installed CLI would fail at import time"
    )


def test_evals_entrypoints_are_importable_as_modules() -> None:
    """The eval CLI adapters spawn these with ``-m``, not by file path."""
    for module in ("evals.run_evals", "evals.compare"):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", module, "--help"],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, f"{module} is not runnable with -m: {result.stderr!r}"


def test_cli_adapters_do_not_resolve_evals_by_file_path() -> None:
    """``__file__``-relative traversal into ``evals/`` breaks in an installed wheel."""
    offenders = [
        module.relative_to(REPO_ROOT)
        for module in (REPO_ROOT / "localrag").rglob("*.py")
        if '"evals"' in module.read_text(encoding="utf-8")
    ]
    assert not offenders, f"{offenders} build a filesystem path to evals/; use `-m` instead"


@pytest.mark.integration
def test_built_wheel_contains_evals_and_runs(tmp_path: Path) -> None:
    """End-to-end proof: build, install into a clean venv, run the CLI."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required to build the wheel")

    dist = tmp_path / "dist"
    subprocess.run(  # noqa: S603
        [uv, "build", "--wheel", "-o", str(dist)], cwd=REPO_ROOT, check=True, capture_output=True
    )
    wheel = next(dist.glob("*.whl"))

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    assert any(name.startswith("evals/") for name in names), "evals/ missing from the wheel"

    venv = tmp_path / "venv"
    subprocess.run([uv, "venv", str(venv)], check=True, capture_output=True)  # noqa: S603
    python = venv / "bin" / "python"
    subprocess.run(  # noqa: S603
        [uv, "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )

    # Run from outside the repo so the source tree cannot satisfy the import.
    result = subprocess.run(  # noqa: S603
        [str(venv / "bin" / "localrag"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, f"installed CLI failed: {result.stderr}"

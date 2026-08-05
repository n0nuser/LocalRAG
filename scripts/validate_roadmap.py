"""Validate ROADMAP.md references against live GitHub data or a JSON fixture."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = "n0nuser/LocalRAG"
ROADMAP = Path(__file__).parents[1] / "ROADMAP.md"
MILESTONE_LINK = re.compile(r"https://github\.com/n0nuser/LocalRAG/milestone/(\d+)")
ISSUE_LINK = re.compile(r"https://github\.com/n0nuser/LocalRAG/issues/(\d+)")
MILESTONE_HEADING = re.compile(
    r"^### \[([^]]+)\]\(https://github\.com/n0nuser/LocalRAG/milestone/(\d+)\)$",
    re.MULTILINE,
)


def github_json(endpoint: str) -> list[dict[str, Any]]:
    result = subprocess.run(  # noqa: S603 - endpoint is constructed below.
        ["gh", "api", "--paginate", endpoint],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def validate(data: dict[str, Any], roadmap: str) -> list[str]:  # noqa: C901
    errors: list[str] = []
    milestones = {item["number"]: item for item in data["milestones"]}
    issues = {item["number"]: item for item in data["issues"]}
    expected = set(milestones)
    linked_milestones = [int(number) for number in MILESTONE_LINK.findall(roadmap)]
    if set(linked_milestones) != expected or len(linked_milestones) != len(expected):
        errors.append("ROADMAP.md must link each live milestone exactly once")

    headings = list(MILESTONE_HEADING.finditer(roadmap))
    heading_numbers = [int(match.group(2)) for match in headings]
    if set(heading_numbers) != expected or len(heading_numbers) != len(expected):
        errors.append("ROADMAP.md must contain one heading for each live milestone")

    for match in headings:
        number = int(match.group(2))
        if number not in milestones:
            errors.append(f"missing live milestone {number}")
        elif match.group(1) != milestones[number]["title"]:
            errors.append(f"milestone {number} title does not match GitHub")

    linked_issues = {int(number) for number in ISSUE_LINK.findall(roadmap)}
    for number in linked_issues:
        if number not in issues:
            errors.append(f"missing live issue {number}")

    for index, match in enumerate(headings):
        section_end = headings[index + 1].start() if index + 1 < len(headings) else len(roadmap)
        section = roadmap[match.end() : section_end]
        milestone_number = int(match.group(2))
        issue_text = section.split("- **Issues:**", 1)[-1].split("- **Hard dependencies:", 1)[0]
        for issue_number in {int(value) for value in ISSUE_LINK.findall(issue_text)}:
            issue = issues.get(issue_number)
            actual = (issue or {}).get("milestone") or {}
            if issue and actual.get("number") != milestone_number:
                errors.append(
                    f"issue {issue_number} is listed under milestone {milestone_number}, "
                    f"not {actual.get('number')}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, help="JSON containing milestones and issues arrays")
    parser.add_argument("--roadmap", type=Path, default=ROADMAP)
    args = parser.parse_args()
    data = json.loads(
        args.fixture.read_text()
        if args.fixture
        else json.dumps(
            {
                "milestones": github_json(f"repos/{REPO}/milestones?state=all"),
                "issues": github_json(f"repos/{REPO}/issues?state=all&per_page=100"),
            }
        )
    )
    errors = validate(data, args.roadmap.read_text())
    if errors:
        sys.stdout.write("\n".join(f"ERROR: {error}" for error in errors) + "\n")
        return 1
    sys.stdout.write("ROADMAP.md references are valid\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

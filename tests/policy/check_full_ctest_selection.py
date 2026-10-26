#!/usr/bin/env python3
"""Verify that full CTest selection loses no tests and runs shared work once."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXPECTED_SHARED = {
    "math.qualified_baseline_staging_policy",
    "math.algorithm_qualification_policy",
    "math.algorithm_comparison",
}


def selected(build: Path, selector: str, pattern: str) -> dict[str, dict[str, object]]:
    command = [
        "ctest",
        "--test-dir",
        str(build),
        "--show-only=json-v1",
        selector,
        pattern,
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout)
    return {test["name"]: test for test in payload["tests"]}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_full_ctest_selection.py BUILD_DIR", file=sys.stderr)
        return 2
    build = Path(sys.argv[1]).resolve(strict=True)
    all_tests = selected(build, "-R", ".*")
    fast = selected(build, "-LE", "^ci-extended$")
    full_normal = selected(build, "-LE", "^ci-once-per-full$")
    shared = selected(build, "-L", "^ci-once-per-full$")

    if not all_tests:
        raise SystemExit("no CTest tests were registered")
    if set(shared) != EXPECTED_SHARED:
        raise SystemExit(
            "shared test selection differs from the approved names: "
            f"expected {sorted(EXPECTED_SHARED)}, got {sorted(shared)}"
        )
    for name, test in shared.items():
        labels = {
            label
            for prop in test.get("properties", [])
            if prop.get("name") == "LABELS"
            for label in prop.get("value", [])
        }
        if "ci-extended" not in labels:
            raise SystemExit(f"shared test is not excluded from fast runs: {name}")
    if set(full_normal) & set(shared):
        raise SystemExit("a shared test would run in both a compiler lane and the shared lane")
    if set(full_normal) | set(shared) != set(all_tests):
        missing = sorted(set(all_tests) - (set(full_normal) | set(shared)))
        unknown = sorted((set(full_normal) | set(shared)) - set(all_tests))
        raise SystemExit(f"full selection mismatch; missing={missing}, unknown={unknown}")
    if not set(fast) <= set(full_normal):
        raise SystemExit("fast selection contains a test excluded from full normal lanes")
    if set(fast) & set(shared):
        raise SystemExit("a shared test would run during fast feedback")

    print(
        "PASS: CTest inventory is complete; "
        f"{len(all_tests)} registered tests, {len(full_normal)} per full lane, "
        f"{len(shared)} once per full run, {len(fast)} fast tests."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

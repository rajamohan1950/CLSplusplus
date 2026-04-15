#!/usr/bin/env python3
"""Waitlist lifecycle test runner — publishes JSON results.

Invokes pytest on tests/test_waitlist.py via a custom plugin that captures
each test's pass/fail state, duration, and error message, then writes:

    website/tests/waitlist/{YYYYMMDDTHHMMSS}.json   — full run
    website/tests/waitlist/manifest.json            — rolling index (last 50)

Usage:
    python3 scripts/run_waitlist_tests.py [--quiet]

Returns exit code 0 iff every test passes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "website" / "tests" / "waitlist"
MANIFEST = RESULTS_DIR / "manifest.json"
TEST_FILE = REPO_ROOT / "tests" / "test_waitlist.py"


class _ResultCollector:
    """pytest plugin — captures per-test results into a flat list."""

    def __init__(self):
        self.cases: list[dict] = []
        self._started: dict[str, float] = {}

    def pytest_runtest_logstart(self, nodeid, location):
        self._started[nodeid] = time.perf_counter()

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and not (
            report.when == "setup" and report.outcome == "failed"
        ):
            return

        nodeid = report.nodeid
        short_name = nodeid.split("::")[-1]
        case_id = _extract_case_id(short_name)
        category = _extract_category(nodeid)
        runtime_ms = (
            (time.perf_counter() - self._started.get(nodeid, time.perf_counter()))
            * 1000
        )
        status = "pass" if report.outcome == "passed" else (
            "skip" if report.outcome == "skipped" else "fail"
        )
        error = ""
        if status == "fail":
            error = str(report.longrepr).split("\n")[-1][:500] if report.longrepr else ""
        self.cases.append(
            {
                "id": case_id,
                "name": short_name,
                "category": category,
                "status": status,
                "runtime_ms": round(runtime_ms, 2),
                "error": error,
            }
        )


def _extract_case_id(name: str) -> str:
    # Test names look like: test_WL_001_stats_empty_returns_seeded_baseline
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "test" and parts[1] == "WL":
        return f"WL-{parts[2]}"
    return name


def _extract_category(nodeid: str) -> str:
    # tests/test_waitlist.py::TestStats::test_WL_001_stats_empty...
    try:
        cls = nodeid.split("::")[1]
        return cls.removeprefix("Test") if hasattr(str, "removeprefix") else cls.replace("Test", "", 1)
    except Exception:
        return "Unknown"


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    collector = _ResultCollector()
    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%dT%H%M%S")

    args = [
        str(TEST_FILE),
        "-p", "no:cacheprovider",
        "--no-header",
        "-q" if quiet else "-v",
    ]
    started_perf = time.perf_counter()
    exit_code = pytest.main(args, plugins=[collector])
    total_runtime_ms = (time.perf_counter() - started_perf) * 1000

    total = len(collector.cases)
    passed = sum(1 for c in collector.cases if c["status"] == "pass")
    failed = sum(1 for c in collector.cases if c["status"] == "fail")
    skipped = sum(1 for c in collector.cases if c["status"] == "skip")
    pass_rate = (passed / total) if total else 0.0

    result_payload = {
        "run_id": run_id,
        "run_timestamp": started_at.isoformat(),
        "suite": "waitlist_lifecycle",
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(pass_rate, 4),
        "total_runtime_ms": round(total_runtime_ms, 1),
        "pytest_exit_code": int(exit_code),
        "cases": collector.cases,
    }

    result_file = RESULTS_DIR / f"{run_id}.json"
    result_file.write_text(json.dumps(result_payload, indent=2))

    # Manifest = rolling index of last 50 runs, newest first
    manifest: list[dict] = []
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text())
        except Exception:
            manifest = []
    manifest.insert(
        0,
        {
            "run_id": run_id,
            "run_timestamp": started_at.isoformat(),
            "file": result_file.name,
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 4),
            "total_runtime_ms": round(total_runtime_ms, 1),
        },
    )
    manifest = manifest[:50]
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    if not quiet:
        print(
            f"\n→ Waitlist suite: {passed}/{total} passed "
            f"({round(pass_rate * 100, 1)}%) in {round(total_runtime_ms)}ms"
        )
        print(f"→ Result: {result_file.relative_to(REPO_ROOT)}")

    return 0 if failed == 0 and exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

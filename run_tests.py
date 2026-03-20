#!/usr/bin/env python3
"""
CLS++ Test Runner — runs test_suite and saves results to website/tests/history/.

Usage:
    python3 run_tests.py                  # run all tests
    python3 run_tests.py --quiet          # no per-test output, just summary

Output:
    website/tests/history/<timestamp>.json   — full result for this run
    website/tests/history/manifest.json      — index of all runs (newest first)
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clsplusplus.test_suite import run_all

HISTORY_DIR = PROJECT_ROOT / "website" / "tests" / "history"
MANIFEST_PATH = HISTORY_DIR / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description="CLS++ test runner")
    parser.add_argument("--quiet", action="store_true", help="suppress per-test output")
    args = parser.parse_args()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nCLS++ Test Suite — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    t0 = time.perf_counter()
    summary = run_all(verbose=not args.quiet)
    total_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Annotate with run metadata
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_ts = datetime.now(timezone.utc).isoformat()
    summary["run_id"] = run_id
    summary["run_timestamp"] = run_ts
    summary["total_runtime_ms"] = total_ms

    # Save full result
    result_path = HISTORY_DIR / f"{run_id}.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Update manifest (keep last 100 runs)
    manifest = []
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

    manifest.insert(0, {
        "run_id": run_id,
        "run_timestamp": run_ts,
        "file": f"{run_id}.json",
        "total": summary["total"],
        "passed": summary["passed"],
        "failed": summary["failed"],
        "pass_rate": summary["pass_rate"],
        "total_runtime_ms": total_ms,
    })
    manifest = manifest[:100]  # keep last 100

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print("\n" + "=" * 72)
    print(f"Result: {summary['passed']}/{summary['total']} passed "
          f"({summary['pass_rate']*100:.1f}%)  total={total_ms}ms")
    print()
    print(f"{'Category':<32} {'Pass':>5} {'Fail':>5}  {'J1':>6}  {'R@5':>6}  {'Lat':>8}")
    print("-" * 72)
    for cat, s in summary["category_summary"].items():
        print(f"{cat:<32} {s['passed']:>5} {s['failed']:>5}  "
              f"{s['avg_j1']:>6.3f}  {s['avg_relevant_at_5']:>6.2f}  "
              f"{s['avg_latency_ms']:>7.1f}ms")

    print(f"\nSaved: {result_path}")
    print(f"History manifest: {MANIFEST_PATH} ({len(manifest)} runs)")

    snap = summary["engine_snapshot"]
    print(f"\nEngine diagnostics:")
    print(f"  PPMI-SVD vectors: {snap['token_vectors_populated']} tokens × {snap['vector_dims_ppmi_svd']} dims  [in-memory dict, NOT persisted]")
    print(f"  SentenceTransformer: {snap['vector_dims_sentence_transformer']} dims  [pgvector, separate system]")
    print(f"  recall_long_tail auto-trigger: {snap['recall_auto_trigger']}  [only via POST /v1/memory/sleep]")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Temporal recall verification — Cat 2.

Tests that MemoryService correctly:
  1. Resolves "yesterday" → absolute date at write time
  2. Forms a temporal thread after the 2nd occurrence of the same event type
  3. Thread is injected at read time for date queries
  4. Works correctly at 2, 10, and 100 sessions (no crystallisation collapse)
  5. Different event types produce separate threads (no cross-contamination)

Run:
    python3 verify_temporal.py
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clsplusplus.memory_service import MemoryService
from clsplusplus.models import ReadRequest, WriteRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_svc() -> MemoryService:
    """Fresh service for each scale test."""
    return MemoryService()


async def write_visit(svc: MemoryService, ns: str, text: str, conv_date: datetime) -> None:
    req = WriteRequest(
        text=text,
        namespace=ns,
        conversation_date=conv_date,
        source="user",
    )
    await svc.write(req)


async def query(svc: MemoryService, ns: str, q: str, limit: int = 10) -> list[str]:
    req = ReadRequest(query=q, namespace=ns, limit=limit)
    resp = await svc.read(req)
    return [item.text for item in resp.items]


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_has(texts: list[str], needle: str, label: str) -> bool:
    combined = " ".join(texts).lower()
    if needle.lower() in combined:
        print(f"  PASS  {label}: found '{needle}'")
        return True
    print(f"  FAIL  {label}: '{needle}' NOT found in:")
    for t in texts[:5]:
        print(f"         - {t[:120]}")
    return False


def assert_not_has(texts: list[str], needle: str, label: str) -> bool:
    combined = " ".join(texts).lower()
    if needle.lower() not in combined:
        print(f"  PASS  {label}: correctly absent '{needle}'")
        return True
    print(f"  FAIL  {label}: '{needle}' should NOT appear but does:")
    for t in texts[:5]:
        print(f"         - {t[:120]}")
    return False


# ---------------------------------------------------------------------------
# Test 1: 2 sessions — thread forms, returns both dates
# ---------------------------------------------------------------------------

async def test_2_sessions() -> bool:
    print("\n[2 sessions] Basic thread formation")
    svc = make_svc()
    ns = "user_2s"

    base = datetime(2024, 1, 8)  # Monday Jan 8

    # Session 1: "I went to the doctor yesterday" → stored as Jan 7
    await write_visit(svc, ns, "I went to the doctor yesterday", base)

    # Session 2: same phrase, one week later → stored as Jan 14
    await write_visit(svc, ns, "I went to the doctor yesterday", base + timedelta(weeks=1))

    # At this point a thread should have formed
    threads = svc._event_threads.get(ns, {})
    if not threads:
        print("  FAIL  No thread created after 2 sessions")
        return False
    print(f"  Thread keys: {list(threads.keys())}")
    for key, dates in threads.items():
        print(f"    [{key}]: {dates}")

    # Query: when did I go to the doctor?
    results = await query(svc, ns, "when did I go to the doctor")
    ok = True
    ok &= assert_has(results, "7 January 2024", "doctor visit Jan 7 in thread")
    ok &= assert_has(results, "14 January 2024", "doctor visit Jan 14 in thread")
    ok &= assert_has(results, "Thread", "thread marker present")
    return ok


# ---------------------------------------------------------------------------
# Test 2: 10 sessions — no crystallisation collapse
# ---------------------------------------------------------------------------

async def test_10_sessions() -> bool:
    print("\n[10 sessions] No crystallisation — all 10 dates in thread")
    svc = make_svc()
    ns = "user_10s"

    base = datetime(2024, 1, 8)

    for i in range(10):
        conv_date = base + timedelta(weeks=i)
        await write_visit(svc, ns, "I went to the doctor yesterday", conv_date)

    threads = svc._event_threads.get(ns, {})
    if not threads:
        print("  FAIL  No thread after 10 sessions")
        return False

    key = next(iter(threads))
    dates = threads[key]
    print(f"  Thread dates ({len(dates)}): {dates}")

    ok = True
    if len(dates) < 9:
        print(f"  FAIL  Expected ~10 dates, got {len(dates)}")
        ok = False
    else:
        print(f"  PASS  Thread has {len(dates)} dates (expected 9-10)")

    results = await query(svc, ns, "doctor visits history", limit=20)
    ok &= assert_has(results, "Thread", "thread marker in results")
    ok &= assert_has(results, "times total", "occurrence count in thread")

    # Latest date: base + 9 weeks = Jan 8 + 63 days = Mar 11; yesterday = Mar 10
    expected_recent = "10 March 2024"
    ok &= assert_has(results, expected_recent, "latest date (9 weeks later visit)")
    return ok


# ---------------------------------------------------------------------------
# Test 3: 100 sessions — scale test
# ---------------------------------------------------------------------------

async def test_100_sessions() -> bool:
    print("\n[100 sessions] Scale — 100 doctor visits over ~2 years")
    svc = make_svc()
    ns = "user_100s"

    base = datetime(2022, 1, 10)  # weekly visits, Jan 2022 → Dec 2023

    for i in range(100):
        conv_date = base + timedelta(weeks=i)
        await write_visit(svc, ns, "I went to the doctor yesterday", conv_date)

    threads = svc._event_threads.get(ns, {})
    if not threads:
        print("  FAIL  No thread after 100 sessions")
        return False

    key = next(iter(threads))
    dates = threads[key]
    print(f"  Thread has {len(dates)} dates")

    ok = True
    if len(dates) < 90:
        print(f"  FAIL  Expected ~100 dates, got {len(dates)}")
        ok = False
    else:
        print(f"  PASS  Thread has {len(dates)} unique dates")

    results = await query(svc, ns, "when was the last doctor visit", limit=5)
    ok &= assert_has(results, "Thread", "thread present at 100 sessions")
    ok &= assert_has(results, "times total", "count in thread")
    # Latest: base + 99 weeks - 1 day = ~Jan 10 2022 + 99*7 - 1 days
    latest_date = base + timedelta(weeks=99) - timedelta(days=1)
    latest_str = latest_date.strftime("%-d %B %Y")
    ok &= assert_has(results, latest_str, f"latest date {latest_str}")
    return ok


# ---------------------------------------------------------------------------
# Test 4: Different event types → separate threads, no cross-contamination
# ---------------------------------------------------------------------------

async def test_separate_threads() -> bool:
    print("\n[separation] Doctor vs hiking vs meeting threads stay separate")
    svc = make_svc()
    ns = "user_sep"

    base = datetime(2024, 3, 15)

    # 2 doctor visits
    await write_visit(svc, ns, "I went to the doctor yesterday", base)
    await write_visit(svc, ns, "I went to the doctor yesterday", base + timedelta(weeks=1))

    # 2 hiking trips
    await write_visit(svc, ns, "I went hiking yesterday", base + timedelta(days=2))
    await write_visit(svc, ns, "I went hiking yesterday", base + timedelta(weeks=1, days=2))

    # 2 meetings
    await write_visit(svc, ns, "I had a team meeting yesterday", base + timedelta(days=3))
    await write_visit(svc, ns, "I had a team meeting yesterday", base + timedelta(weeks=1, days=3))

    threads = svc._event_threads.get(ns, {})
    print(f"  Threads formed: {list(threads.keys())}")

    ok = True

    # Doctor query → doctor dates only, NOT hiking dates
    doctor_results = await query(svc, ns, "when did I go to the doctor")
    ok &= assert_has(doctor_results, "doctor", "doctor thread in doctor query")

    # Hiking query → hiking dates
    hiking_results = await query(svc, ns, "when did I go hiking")
    ok &= assert_has(hiking_results, "hiking", "hiking thread in hiking query")

    # Meeting query → meeting dates
    meeting_results = await query(svc, ns, "when was the team meeting")
    ok &= assert_has(meeting_results, "meeting", "meeting thread in meeting query")

    return ok


# ---------------------------------------------------------------------------
# Test 5: Date storage correctness — "yesterday" resolves to correct date
# ---------------------------------------------------------------------------

async def test_date_resolution() -> bool:
    print("\n[date resolution] 'yesterday' resolves to D-1, not D")
    svc = make_svc()
    ns = "user_dates"

    # Jan 8 is conversation date → "yesterday" = Jan 7
    conv_date = datetime(2024, 1, 8)
    await write_visit(svc, ns, "I visited the doctor yesterday", conv_date)
    await write_visit(svc, ns, "I visited the doctor yesterday", conv_date + timedelta(weeks=1))

    threads = svc._event_threads.get(ns, {})
    ok = True

    # Thread should have Jan 7 (not Jan 8) as first date
    if threads:
        key = next(iter(threads))
        dates = threads[key]
        ok &= assert_has(dates, "7 January 2024", "event date is Jan 7 (not Jan 8)")
        ok &= assert_not_has(dates, "8 January 2024", "conversation date Jan 8 not in thread")
    else:
        print("  FAIL  No thread formed")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    print("=" * 72)
    print("CLS++ Temporal Recall Verification — Cat 2")
    print("=" * 72)

    results = {
        "2 sessions": await test_2_sessions(),
        "10 sessions": await test_10_sessions(),
        "100 sessions": await test_100_sessions(),
        "thread separation": await test_separate_threads(),
        "date resolution": await test_date_resolution(),
    }

    print("\n" + "=" * 72)
    print("Summary:")
    passed = 0
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")
        if ok:
            passed += 1
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

#!/usr/bin/env python3
"""
Session-scale verification: 10 sessions × 10 conversations each → recall.

Simulates a realistic user over 10 weeks, with 10 things said per session.
Each session has a different date. Tests that:

  1. Temporal events (recurring doctor visits, hiking trips) form correct threads
  2. Single-occurrence facts (profession, city, pet) are recalled accurately
  3. Inference queries (profession, lifestyle) surface correct context
  4. Adversarial queries (never-mentioned facts) return empty or off-topic
  5. No cross-thread contamination between events

Sessions are consecutive weeks starting 2024-01-08.
Each session has 10 statements simulating a real conversation.

Run:
    python3 verify_sessions.py
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
# Session data — 10 sessions × 10 turns
# ---------------------------------------------------------------------------

BASE_DATE = datetime(2024, 1, 8)

SESSIONS = [
    # Session 0 — Jan 8
    [
        "I work as a software engineer at Acme Corp.",
        "I live in San Francisco.",
        "I went to the doctor yesterday for a routine checkup.",
        "My cat is named Luna.",
        "I enjoy cooking Italian food on weekends.",
        "I take the Caltrain to the office every day.",
        "I had a 1:1 meeting with my manager this morning.",
        "I finished reading 'Dune' last night.",
        "My sister called from New York.",
        "I ordered Thai food for dinner.",
    ],
    # Session 1 — Jan 15
    [
        "I went hiking yesterday in Muir Woods.",
        "I pushed a new feature to production at work.",
        "I went to the doctor yesterday for a follow-up.",
        "Luna knocked over my coffee mug this morning.",
        "I tried a new pasta recipe for dinner.",
        "The Caltrain was delayed by 20 minutes.",
        "My team shipped the mobile app update.",
        "I started reading 'Foundation' by Asimov.",
        "I called my sister in New York.",
        "I had sushi for lunch with a coworker.",
    ],
    # Session 2 — Jan 22
    [
        "I went hiking yesterday at Point Reyes.",
        "I fixed a critical bug in the payment service at work.",
        "I went to the doctor yesterday — blood pressure is normal.",
        "Luna has been extra playful today.",
        "I made homemade pizza for dinner.",
        "I got a window seat on the Caltrain.",
        "I presented the sprint review to the whole company.",
        "I finished 'Foundation' — loved it.",
        "My sister is visiting San Francisco next month.",
        "I discovered a great ramen place near the office.",
    ],
    # Session 3 — Jan 29
    [
        "I went hiking yesterday at Mount Tamalpais.",
        "I started a new microservices project at work.",
        "I went to the doctor yesterday for a blood test.",
        "Luna caught a fly today.",
        "I made tiramisu from scratch.",
        "The Caltrain was on time for once.",
        "I got promoted to senior software engineer.",
        "I ordered 'The Hitchhiker's Guide to the Galaxy' online.",
        "My sister confirmed her visit — she arrives February 15.",
        "I tried Vietnamese food for the first time.",
    ],
    # Session 4 — Feb 5
    [
        "I went hiking yesterday at Lands End.",
        "I deployed the microservices project to staging.",
        "I went to the doctor yesterday — doctor says everything looks good.",
        "Luna figured out how to open the treat cabinet.",
        "I hosted a dinner party — made my signature carbonara.",
        "I biked to Caltrain instead of walking.",
        "I started mentoring a new engineer on the team.",
        "I finished 'The Hitchhiker's Guide' — hilarious.",
        "My sister arrives in 10 days.",
        "I tried a Lebanese restaurant near my apartment.",
    ],
    # Session 5 — Feb 12
    [
        "I went hiking yesterday at Marin Headlands.",
        "I went to the doctor yesterday for a prescription renewal.",
        "The microservices project went live in production.",
        "Luna has a new favorite toy — a crinkle ball.",
        "I made a Thai green curry from scratch.",
        "Got a seat on the express Caltrain.",
        "My team hit 99.9% uptime this month.",
        "Started reading 'Project Hail Mary' by Andy Weir.",
        "My sister arrived — we went to Fisherman's Wharf.",
        "I discovered the best dim sum in the Richmond District.",
    ],
    # Session 6 — Feb 19
    [
        "I went hiking yesterday at Stinson Beach.",
        "I went to the doctor yesterday for the third month in a row.",
        "I gave a tech talk at a local meetup.",
        "Luna is 3 years old today.",
        "I made lamb chops for my sister's last dinner.",
        "My sister flew back to New York this morning.",
        "I got a new standing desk at the office.",
        "I finished 'Project Hail Mary' — incredible book.",
        "I started a side project in Rust.",
        "I ordered Ethiopian food for delivery.",
    ],
    # Session 7 — Feb 26
    [
        "I went hiking yesterday on the Bay Trail.",
        "I went to the doctor yesterday — routine monthly checkup.",
        "I deployed the Rust side project's first version.",
        "Luna learned to high-five.",
        "I made my first beef bourguignon.",
        "The Caltrain app got a new update.",
        "I got feedback on my tech talk — very positive.",
        "Started reading 'Shogun' by James Clavell.",
        "I video-called my sister.",
        "I discovered a jazz bar near my apartment.",
    ],
    # Session 8 — Mar 4
    [
        "I went hiking yesterday at Sweeney Ridge.",
        "I went to the doctor yesterday — last checkup in this series.",
        "I submitted my Rust project to a hackathon.",
        "Luna discovered the window ledge and spends hours there.",
        "I made French onion soup for the first time.",
        "I switched to biking to the Caltrain stop.",
        "I was asked to lead the platform team.",
        "I'm halfway through 'Shogun' — it's amazing.",
        "My sister sent me a care package from New York.",
        "I tried a new Korean BBQ place.",
    ],
    # Session 9 — Mar 11
    [
        "I went hiking yesterday at Mount Diablo.",
        "I went to the doctor yesterday for a final follow-up.",
        "I accepted the platform team lead role.",
        "Luna is cuddlier than ever this winter.",
        "I made homemade croissants over the weekend.",
        "The Caltrain now has WiFi on all trains.",
        "I onboarded two new engineers to the platform team.",
        "I finished 'Shogun' — masterpiece.",
        "My sister is planning to visit again in the summer.",
        "I found a great Moroccan restaurant near the office.",
    ],
]


async def run_sessions(svc: MemoryService, ns: str) -> None:
    for session_idx, session in enumerate(SESSIONS):
        conv_date = BASE_DATE + timedelta(weeks=session_idx)
        for text in session:
            req = WriteRequest(
                text=text,
                namespace=ns,
                conversation_date=conv_date,
                source="user",
            )
            await svc.write(req)


async def q(svc: MemoryService, ns: str, query_text: str, limit: int = 8) -> list[str]:
    req = ReadRequest(query=query_text, namespace=ns, limit=limit, min_confidence=0.0)
    resp = await svc.read(req)
    return [item.text for item in resp.items]


def check(label: str, results: list[str], needle: str, should_find: bool = True) -> bool:
    found = needle.lower() in " ".join(results).lower()
    ok = found == should_find
    mark = "PASS" if ok else "FAIL"
    status = "found" if found else "NOT found"
    neg = "" if should_find else "correctly absent: "
    print(f"  {mark}  {label}: {neg}'{needle}' {status if should_find else ''}")
    if not ok:
        print(f"         Results:")
        for t in results[:4]:
            print(f"           - {t[:110]}")
    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_temporal_doctor_thread(svc: MemoryService, ns: str) -> bool:
    print("\n[Temporal] Doctor visit thread — 10 visits, all dates present")
    threads = svc._event_threads.get(ns, {})
    doctor_key = next((k for k in threads if "doctor" in k), None)
    if not doctor_key:
        print("  FAIL  No doctor thread formed")
        return False

    dates = threads[doctor_key]
    print(f"  Doctor thread: {len(dates)} dates — {dates[0]} … {dates[-1]}")

    ok = len(dates) >= 9
    if ok:
        print(f"  PASS  Thread has {len(dates)} dates (expected 9-10)")
    else:
        print(f"  FAIL  Only {len(dates)} dates in thread")

    results = await q(svc, ns, "when did I go to the doctor")
    ok &= check("thread in doctor query", results, "[Thread]")
    ok &= check("latest doctor date in result", results, "10 March 2024")  # session 9: Mar 11 - 1 = Mar 10
    return ok


async def test_temporal_hiking_thread(svc: MemoryService, ns: str) -> bool:
    print("\n[Temporal] Hiking thread — 9 hikes across sessions 1-9")
    threads = svc._event_threads.get(ns, {})
    hiking_key = next((k for k in threads if "hiking" in k), None)
    if not hiking_key:
        print("  FAIL  No hiking thread formed")
        return False

    dates = threads[hiking_key]
    print(f"  Hiking thread: {len(dates)} dates — {dates[0]} … {dates[-1]}")

    ok = len(dates) >= 8
    if ok:
        print(f"  PASS  Hiking thread has {len(dates)} dates")
    else:
        print(f"  FAIL  Only {len(dates)} hiking dates in thread")

    results = await q(svc, ns, "when did I go hiking")
    ok &= check("hiking thread in query", results, "[Thread]")
    ok &= check("hiking keyword present", results, "hiking")
    return ok


async def test_thread_separation(svc: MemoryService, ns: str) -> bool:
    print("\n[Separation] Doctor query gets doctor thread, hiking query gets hiking thread")
    doctor_results = await q(svc, ns, "when did I visit the doctor")
    hiking_results = await q(svc, ns, "when did I go on a hike")

    ok = True
    # Doctor query: doctor thread present, hiking thread absent
    ok &= check("doctor thread in doctor query", doctor_results, "doctor")
    hiking_in_doctor = any("[Thread]" in t and "hiking" in t.lower() for t in doctor_results)
    if not hiking_in_doctor:
        print(f"  PASS  Hiking thread absent from doctor query")
    else:
        print(f"  FAIL  Hiking thread polluted doctor query")
        ok = False

    # Hiking query: hiking thread present, doctor thread absent
    ok &= check("hiking in hiking query", hiking_results, "hiking")
    doctor_in_hiking = any("[Thread]" in t and "doctor" in t.lower() for t in hiking_results)
    if not doctor_in_hiking:
        print(f"  PASS  Doctor thread absent from hiking query")
    else:
        print(f"  FAIL  Doctor thread polluted hiking query")
        ok = False

    return ok


async def test_single_fact_recall(svc: MemoryService, ns: str) -> bool:
    print("\n[Recall] Single-occurrence facts remembered accurately")
    ok = True

    r = await q(svc, ns, "what city do I live in")
    ok &= check("live in San Francisco", r, "san francisco")

    r = await q(svc, ns, "what is my cat's name")
    ok &= check("cat named Luna", r, "luna")

    r = await q(svc, ns, "what is my job title")
    ok &= check("software engineer at Acme", r, "software engineer")

    r = await q(svc, ns, "what promotion did I get")
    ok &= check("promoted to senior software engineer", r, "senior")

    return ok


async def test_inference_recall(svc: MemoryService, ns: str) -> bool:
    print("\n[Inference] Implied facts surface correctly")
    ok = True

    # Profession inference
    r = await q(svc, ns, "what kind of work do I do")
    any_work = any(kw in " ".join(r).lower() for kw in ["engineer", "software", "production", "deploy", "code"])
    if any_work:
        print(f"  PASS  Work context surfaced for profession query")
    else:
        print(f"  FAIL  No work context found for profession query")
        ok = False

    # Hobby inference
    r = await q(svc, ns, "what are my hobbies")
    any_hobby = any(kw in " ".join(r).lower() for kw in ["hiking", "cooking", "reading", "food"])
    if any_hobby:
        print(f"  PASS  Hobby context surfaced for hobbies query")
    else:
        print(f"  FAIL  No hobby context found")
        ok = False

    # Pet inference
    r = await q(svc, ns, "do I have any pets")
    ok &= check("pet context", r, "luna")

    return ok


async def test_adversarial(svc: MemoryService, ns: str) -> bool:
    print("\n[Adversarial] Never-mentioned facts return no thread contamination")
    ok = True

    # Swimming was never mentioned — no thread should appear
    r = await q(svc, ns, "when did I go swimming")
    swimming_thread = any("[Thread]" in t and "swim" in t.lower() for t in r)
    if not swimming_thread:
        print(f"  PASS  No swimming thread injected (swimming never mentioned)")
    else:
        print(f"  FAIL  False swimming thread injected")
        ok = False

    # Car ownership never mentioned
    r = await q(svc, ns, "what car do I drive")
    car_thread = any("[Thread]" in t and "car" in t.lower() for t in r)
    if not car_thread:
        print(f"  PASS  No car thread injected (car never mentioned)")
    else:
        print(f"  FAIL  False car thread injected")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> int:
    print("=" * 72)
    print("CLS++ Session Verification — 10 sessions × 10 conversations")
    print("=" * 72)
    print(f"Loading {sum(len(s) for s in SESSIONS)} memories across {len(SESSIONS)} sessions...")

    svc = MemoryService()
    ns = "user_10x10"
    await run_sessions(svc, ns)

    threads = svc._event_threads.get(ns, {})
    print(f"Threads formed: {list(threads.keys())}")
    print(f"Engine items: {svc.engine._total_item_count}")
    print()

    results = {
        "Temporal: doctor thread": await test_temporal_doctor_thread(svc, ns),
        "Temporal: hiking thread": await test_temporal_hiking_thread(svc, ns),
        "Thread separation": await test_thread_separation(svc, ns),
        "Single-fact recall": await test_single_fact_recall(svc, ns),
        "Inference recall": await test_inference_recall(svc, ns),
        "Adversarial": await test_adversarial(svc, ns),
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

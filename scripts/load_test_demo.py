#!/usr/bin/env python3
"""
Load test: 5 concurrent users, 5-min chat + criss-cross.
Each user: chats with Claude/Gemini/OpenAI, then asks cross-model.
Tests the same API the UI uses.

Usage:
  python scripts/load_test_demo.py           # 5 concurrent users
  python scripts/load_test_demo.py 1         # 1 user (smoke test)
  python scripts/load_test_demo.py 5 seq     # 5 users sequential
"""
import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Tuple

try:
    import httpx
except ImportError:
    print("pip install httpx")
    raise

API_URL = "https://clsplusplus-api.onrender.com"
TIMEOUT = 120.0  # Cold start can take 60-90s
DELAY_BETWEEN_REQUESTS = 3.0  # Avoid overwhelming free tier; 5 users * 11 req = 55 req


@dataclass
class UserResult:
    user_id: int
    namespace: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    facts_told: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)


async def chat(user_id: int, namespace: str, model: str, message: str) -> Tuple[bool, str]:
    """Send message to model, return (success, reply)."""
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                f"{API_URL}/v1/demo/chat",
                json={"model": model, "message": message, "namespace": namespace},
            )
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
            data = r.json()
            reply = data.get("reply", "")
            if "Add " in reply and "env" in reply:
                return False, reply
            if "error" in reply.lower():
                return False, reply
            return True, reply
    except Exception as e:
        return False, str(e)[:200]


async def run_user(user_id: int) -> UserResult:
    """One user: chat 5 min equivalent + criss-cross."""
    ns = f"loadtest-u{user_id}-{int(time.time())}"
    r = UserResult(user_id=user_id, namespace=ns, ok=True)

    facts = {
        "name": f"User{user_id}",
        "pet": f"dog named Spot{user_id}",
        "food": f"pizza with olives",
        "city": f"Boston",
        "hobby": f"chess",
    }

    # Phase 1: Tell each model 2 facts (simulates ~5 min of chat)
    statements = [
        ("claude", f"My name is {facts['name']}."),
        ("claude", f"I have a {facts['pet']}."),
        ("openai", f"My favorite food is {facts['food']}."),
        ("openai", f"I live in {facts['city']}."),
        ("gemini", f"My hobby is {facts['hobby']}."),
        ("gemini", f"My name is {facts['name']} and I love {facts['food']}."),
    ]
    r.facts_told = facts

    for model, msg in statements:
        ok, reply = await chat(user_id, ns, model, msg)
        if not ok:
            r.ok = False
            r.errors.append(f"{model}: {reply[:80]}")
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    # Phase 2: Criss-cross — ask in different model than told
    questions = [
        ("openai", "What is my name?", "name"),
        ("gemini", "What is my pet?", "pet"),
        ("claude", "What is my favorite food?", "food"),
        ("claude", "What city do I live in?", "city"),
        ("openai", "What is my hobby?", "hobby"),
    ]

    for model, q, key in questions:
        ok, reply = await chat(user_id, ns, model, q)
        r.answers[key] = reply
        if not ok:
            r.ok = False
            r.errors.append(f"{model} Q: {reply[:80]}")
        else:
            expected = facts[key].lower()
            if expected not in reply.lower():
                r.ok = False
                r.errors.append(f"{model} answer missing '{expected}' in: {reply[:60]}")
        await asyncio.sleep(DELAY_BETWEEN_REQUESTS)

    return r


async def main():
    n_users = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sequential = len(sys.argv) > 2 and sys.argv[2].lower() == "seq"

    print(f"Load test: {n_users} user(s), {'sequential' if sequential else 'concurrent'}, chat + criss-cross")
    print(f"API: {API_URL}")
    print("Starting...")
    start = time.time()

    if sequential:
        results = []
        for i in range(n_users):
            r = await run_user(i)
            results.append(r)
    else:
        results = await asyncio.gather(*[run_user(i) for i in range(n_users)])

    elapsed = time.time() - start
    passed = sum(1 for r in results if r.ok)
    failed = n_users - passed

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Passed: {passed}/{n_users}, Failed: {failed}/{n_users}")

    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"\n  User {r.user_id} ({r.namespace}): {status}")
        if r.errors:
            for e in r.errors[:3]:
                print(f"    - {e}")

    if failed > 0:
        print("\nSome users failed. Ensure CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY, CLS_GOOGLE_API_KEY are set in Render.")
        return 1
    print("\nAll 5 users passed.")
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))

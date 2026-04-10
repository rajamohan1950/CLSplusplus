"""
CLS++ Stress Test — Real API calls, no simulation.
Tests memory across model switches with increasing complexity.

5 test levels:
  1. Single Hop — learn on GPT-4, recall on Claude
  2. Multi Hop — chain facts across 3 models
  3. Adversarial — contradictions, corrections, overrides
  4. Conversational — long multi-turn dialogue
  5. Stress — rapid concurrent writes, large payloads, edge cases
"""

import os
import time
import httpx
from clsplusplus import Brain

CLS = "http://localhost:8181"
OAI_KEY = os.environ.get("OPENAI_API_KEY", "")
ANT_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

if not OAI_KEY or not ANT_KEY:
    print("ERROR: Set OPENAI_API_KEY and ANTHROPIC_API_KEY env vars")
    exit(1)


def call_gpt(user_msg, namespace="stress-test", system="You are a helpful assistant."):
    """Call GPT-4o-mini through CLS++ proxy."""
    resp = httpx.post(f"{CLS}/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    }, headers={
        "Authorization": f"Bearer {OAI_KEY}",
        "X-User-Id": namespace,
    }, timeout=60)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_claude(user_msg, namespace="stress-test", system="You are a helpful assistant."):
    """Call Claude through CLS++ proxy."""
    resp = httpx.post(f"{CLS}/v1/messages", json={
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
    }, headers={
        "x-api-key": ANT_KEY,
        "anthropic-version": "2023-06-01",
        "X-User-Id": namespace,
    }, timeout=60)
    data = resp.json()
    return data["content"][0]["text"]


def check(brain, query, expected_substring, test_name):
    """Check if brain.ask() returns something containing the expected substring."""
    results = brain.ask(query, limit=5)
    found = any(expected_substring.lower() in r.lower() for r in results)
    status = "PASS" if found else "FAIL"
    print(f"  [{status}] {test_name}")
    if not found:
        print(f"         Expected: '{expected_substring}'")
        print(f"         Got: {results[:3]}")
    return found


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: SINGLE HOP — Learn on GPT-4, recall on Claude
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: SINGLE HOP")
print("Learn facts via GPT-4, recall via Claude")
print("=" * 60)

ns = f"single-hop-{int(time.time())}"
brain = Brain(ns, url=CLS)

# Tell GPT-4 some facts
print("\n  Sending to GPT-4...")
gpt_reply = call_gpt("My name is Arjun. I work at Tesla as a robotics engineer. I use ROS2 and C++.", namespace=ns)
print(f"  GPT-4: {gpt_reply[:100]}...")

# Now ask Claude — does it know?
print("\n  Switching to Claude...")
claude_reply = call_claude("What do you know about me? What's my name and where do I work?", namespace=ns)
print(f"  Claude: {claude_reply[:200]}...")

# Verify via Brain SDK
print("\n  Verification via Brain SDK:")
check(brain, "name", "Arjun", "Name recalled")
check(brain, "company", "Tesla", "Company recalled")
check(brain, "programming", "C++", "Language recalled")
print()


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: MULTI HOP — Chain facts across GPT-4 → Claude → GPT-4
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: MULTI HOP")
print("Chain: GPT-4 learns → Claude adds → GPT-4 recalls all")
print("=" * 60)

ns2 = f"multi-hop-{int(time.time())}"
brain2 = Brain(ns2, url=CLS)

# Hop 1: GPT-4 learns the basics
print("\n  Hop 1: GPT-4 learns basics...")
call_gpt("I'm Maya, a data scientist. I use Python and SQL daily.", namespace=ns2)

# Hop 2: Claude learns more context
print("  Hop 2: Claude adds more context...")
call_claude("I just joined Stripe last month. Working on fraud detection.", namespace=ns2)

# Hop 3: Back to GPT-4 — should know EVERYTHING from both hops
print("  Hop 3: Back to GPT-4, asking about everything...")
gpt_final = call_gpt("Summarize everything you know about me — name, company, role, tools, project.", namespace=ns2)
print(f"  GPT-4 final: {gpt_final[:250]}...")

print("\n  Verification:")
check(brain2, "name", "Maya", "Name from Hop 1")
check(brain2, "company", "Stripe", "Company from Hop 2")
check(brain2, "project", "fraud", "Project from Hop 2")
check(brain2, "tools", "Python", "Tools from Hop 1")
print()


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: ADVERSARIAL — Contradictions, corrections, edge cases
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: ADVERSARIAL")
print("Contradictions, corrections, belief updates")
print("=" * 60)

ns3 = f"adversarial-{int(time.time())}"
brain3 = Brain(ns3, url=CLS)

# Set initial fact
brain3.learn("I work at Google as a software engineer")
print("\n  Initial: 'I work at Google'")

# Contradict it
brain3.learn("Actually I just left Google. I now work at Meta.")
print("  Contradiction: 'I left Google, now at Meta'")

# Use brain.correct() for clean update
brain3.correct("I work at Google as a software engineer", "I work at Meta as a senior engineer")
print("  Corrected: Google → Meta")

# Ask — should return Meta, not Google
print("\n  Verification:")
results = brain3.ask("Where do I work?")
has_meta = any("meta" in r.lower() for r in results)
has_google = any("google" in r.lower() and "left" not in r.lower() for r in results[:1])
print(f"  [{'PASS' if has_meta else 'FAIL'}] Returns Meta")
print(f"  [{'PASS' if not has_google else 'FAIL'}] Google not in top result")

# Emoji and special characters
brain3.learn("My favorite emoji is 🚀 and I love café lattes ☕")
check(brain3, "emoji", "🚀", "Unicode/emoji handling")

# Very long fact
long_fact = "I am building a distributed system that processes " + ", ".join([f"step-{i}" for i in range(50)]) + " in a pipeline architecture."
brain3.learn(long_fact)
check(brain3, "distributed system", "pipeline", "Long fact (500+ chars)")

# Empty and tiny inputs
try:
    brain3.learn("")
    print("  [FAIL] Empty string should fail gracefully")
except:
    print("  [PASS] Empty string rejected")

try:
    brain3.learn("hi")
    print("  [INFO] Very short input accepted (may or may not store)")
except:
    print("  [PASS] Very short input rejected")
print()


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: CONVERSATIONAL — Multi-turn dialogue across models
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 4: CONVERSATIONAL")
print("10-turn dialogue, model switch mid-conversation")
print("=" * 60)

ns4 = f"convo-{int(time.time())}"
brain4 = Brain(ns4, url=CLS)

turns = [
    ("gpt", "I'm building a SaaS for healthcare compliance"),
    ("gpt", "We use Django and PostgreSQL with HIPAA compliance"),
    ("gpt", "Our team is 5 engineers based in Austin, Texas"),
    ("gpt", "We just raised a $2M seed round from Y Combinator"),
    ("gpt", "I'm the CTO, my co-founder handles product"),
    # --- SWITCH TO CLAUDE ---
    ("claude", "What tech stack should we consider for scaling?"),
    ("claude", "We're planning to hire 3 more backend engineers"),
    ("claude", "Budget for infrastructure is $15K/month on AWS"),
    ("claude", "We need to support 10K concurrent users by Q3"),
    ("claude", "What do you know about our startup so far?"),
]

print()
for i, (model, msg) in enumerate(turns, 1):
    fn = call_gpt if model == "gpt" else call_claude
    label = "GPT-4" if model == "gpt" else "Claude"
    if i == 6:
        print("  --- MODEL SWITCH: GPT-4 → Claude ---")
    print(f"  Turn {i} [{label}]: {msg[:60]}...")
    reply = fn(msg, namespace=ns4)
    print(f"    → {reply[:80]}...")

print("\n  Verification (Claude should know GPT-4's context):")
check(brain4, "funding", "Y Combinator", "YC funding (from GPT-4 turns)")
check(brain4, "city", "Austin", "Location (from GPT-4 turns)")
check(brain4, "stack", "Django", "Tech stack (from GPT-4 turns)")
check(brain4, "role", "CTO", "Role (from GPT-4 turns)")
check(brain4, "budget", "15K", "Budget (from Claude turns)")
check(brain4, "hiring", "backend", "Hiring plans (from Claude turns)")
print()


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: STRESS — Rapid writes, bulk operations, edge cases
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 5: STRESS")
print("Rapid writes, bulk ingestion, retrieval under load")
print("=" * 60)

ns5 = f"stress-{int(time.time())}"
brain5 = Brain(ns5, url=CLS)

# Rapid-fire 20 facts
print("\n  Writing 20 facts rapidly...")
t0 = time.time()
for i in range(20):
    brain5.learn(f"Fact number {i}: The user's preference #{i} is item-{i*7}")
elapsed = time.time() - t0
print(f"  20 writes in {elapsed:.2f}s ({elapsed/20*1000:.0f}ms per write)")

# Bulk retrieval
print("  Retrieving all 20...")
t0 = time.time()
all_facts = brain5.ask("all preferences", limit=20)
elapsed = time.time() - t0
print(f"  Retrieved {len(all_facts)} facts in {elapsed:.2f}s")
print(f"  [{'PASS' if len(all_facts) >= 15 else 'FAIL'}] At least 15 of 20 retrieved")

# Absorb a large document
large_doc = "\n".join([f"Section {i}: The user has requirement {i} which involves component-{i} and dependency-{i}." for i in range(30)])
print(f"\n  Absorbing large document ({len(large_doc)} chars, 30 sections)...")
t0 = time.time()
absorbed = brain5.absorb(large_doc)
elapsed = time.time() - t0
print(f"  Absorbed {absorbed} facts in {elapsed:.2f}s")
print(f"  [{'PASS' if absorbed >= 20 else 'FAIL'}] At least 20 of 30 sections absorbed")

# Concurrent model calls
print("\n  Concurrent: GPT-4 and Claude reading same memory...")
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    f1 = ex.submit(lambda: call_gpt("What are the user's preferences?", namespace=ns5))
    f2 = ex.submit(lambda: call_claude("What requirements does the user have?", namespace=ns5))
    r1 = f1.result()
    r2 = f2.result()
print(f"  GPT-4: {r1[:80]}...")
print(f"  Claude: {r2[:80]}...")
print(f"  [PASS] Both models responded without error")
print()


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)

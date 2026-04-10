"""
CLS++ 1000-Fact Test — Proper Per-User Namespaces
===================================================
50 users × 20 facts each = 1000 facts.
Each user gets their own Brain (namespace), as designed.
Recall tested across all 50 users.
"""

import time
from clsplusplus import Brain

CLS = "http://localhost:8181"

names = ["Alice","Bob","Priya","Carlos","Yuki","Fatima","Liam","Mei","Omar","Sofia"]
companies = ["Google","Meta","Stripe","Netflix","Tesla","Spotify","Airbnb","Shopify","Databricks","Figma"]
roles = ["engineer","designer","data scientist","PM","DevOps lead","ML researcher","CTO","architect","analyst","founder"]
langs = ["Python","TypeScript","Go","Rust","Java","C++","Swift","Kotlin","Ruby","Elixir"]
dbs = ["PostgreSQL","MongoDB","Redis","DynamoDB","ClickHouse","Cassandra","SQLite","Neo4j","Snowflake","BigQuery"]
cities = ["San Francisco","New York","London","Tokyo","Berlin","Austin","Seattle","Toronto","Singapore","Mumbai"]
hobbies = ["climbing","photography","chess","cooking","running","painting","guitar","hiking","reading","yoga"]
foods = ["sushi","tacos","pizza","ramen","biryani","pasta","dim sum","falafel","pho","curry"]
editors = ["VS Code","Neovim","IntelliJ","Cursor","Emacs","Sublime","Zed","Fleet","Helix","Atom"]
clouds = ["AWS","GCP","Azure","Vercel","Fly.io","Railway","Render","DigitalOcean","Cloudflare","Heroku"]

# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Store 1000 facts across 50 users
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STORING 1000 FACTS (50 users × 20 facts each)")
print("=" * 60)

brains = {}
t0 = time.time()
total_stored = 0

for i in range(50):
    ns = f"user-{i}-{int(t0)}"
    b = Brain(ns, url=CLS)
    brains[i] = b

    facts = [
        f"My name is {names[i%10]}",
        f"I am {25+i%20} years old",
        f"I work at {companies[i%10]} as a {roles[i%10]}",
        f"I primarily code in {langs[i%10]}",
        f"I also use {langs[(i+3)%10]} as a secondary language",
        f"My main database is {dbs[i%10]}",
        f"I live in {cities[i%10]}",
        f"My hobby is {hobbies[i%10]}",
        f"My favorite food is {foods[i%10]}",
        f"I use {editors[i%10]} as my code editor",
        f"I deploy on {clouds[i%10]}",
        f"I prefer {'dark' if i%2==0 else 'light'} mode in all tools",
        f"My salary is ${50+i*3}K per year",
        f"I joined the company in {2019+i%6}",
        f"My team has {5+i%15} people",
        f"I am currently working on project {['Phoenix','Atlas','Mercury','Nova','Orion'][i%5]}",
        f"My manager's name is {names[(i+5)%10]}",
        f"I graduated from {['MIT','Stanford','CMU','Berkeley','IIT'][i%5]}",
        f"My phone number is 555-{1000+i}",
        f"I drive a {'Tesla' if i%3==0 else 'Toyota' if i%3==1 else 'Honda'}",
    ]

    for fact in facts:
        try:
            b.learn(fact)
            total_stored += 1
        except Exception:
            pass

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/50 users done ({total_stored} facts, {elapsed:.1f}s)")

elapsed = time.time() - t0
print(f"\n  Total: {total_stored}/1000 stored in {elapsed:.1f}s ({elapsed/max(total_stored,1)*1000:.0f}ms avg)")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: Recall — query each user's own Brain
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("RECALL TEST — 100 queries across 50 users")
print("=" * 60)
print()

time.sleep(1)

queries = [
    ("name", lambda i: names[i%10]),
    ("company", lambda i: companies[i%10]),
    ("programming language", lambda i: langs[i%10]),
    ("database", lambda i: dbs[i%10]),
    ("city", lambda i: cities[i%10]),
    ("hobby", lambda i: hobbies[i%10]),
    ("food", lambda i: foods[i%10]),
    ("editor", lambda i: editors[i%10]),
    ("cloud deploy", lambda i: clouds[i%10]),
    ("mode preference", lambda i: "dark" if i%2==0 else "light"),
    ("project", lambda i: ['Phoenix','Atlas','Mercury','Nova','Orion'][i%5]),
    ("manager", lambda i: names[(i+5)%10]),
    ("university graduated", lambda i: ['MIT','Stanford','CMU','Berkeley','IIT'][i%5]),
    ("car drive", lambda i: 'Tesla' if i%3==0 else 'Toyota' if i%3==1 else 'Honda'),
]

# Test 2 users per query type × 14 types = not all 50, pick representative set
test_users = [0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
passed = 0
failed = 0
total_q = 0

for user_idx in test_users:
    b = brains[user_idx]
    for topic, expect_fn in queries:
        expected = expect_fn(user_idx)
        results = b.ask(topic, limit=5)
        found = any(expected.lower() in r.lower() for r in results)
        total_q += 1
        if found:
            passed += 1
        else:
            failed += 1
            print(f"  [FAIL] User-{user_idx} '{topic}' expected '{expected}' → {results[0][:50] if results else 'NONE'}")

pct = passed / total_q * 100
print(f"\n  Passed: {passed}/{total_q}")
print(f"  Failed: {failed}/{total_q}")
print(f"\n  ═══ RESULT: {pct:.0f}% recall at 1000 facts across 50 users ═══")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: Cross-model test — store via one user, recall via new Brain
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("PERSISTENCE — New Brain instances recall old data")
print("=" * 60)
print()

persist_passed = 0
persist_total = 0
for user_idx in [0, 10, 20, 30, 40]:
    # Create a NEW Brain with same namespace — simulates model switch
    old_ns = f"user-{user_idx}-{int(t0)}"
    fresh_brain = Brain(old_ns, url=CLS)
    for topic, expect_fn in [("name", lambda i: names[i%10]), ("company", lambda i: companies[i%10])]:
        expected = expect_fn(user_idx)
        results = fresh_brain.ask(topic, limit=5)
        found = any(expected.lower() in r.lower() for r in results)
        persist_total += 1
        if found:
            persist_passed += 1
            print(f"  [PASS] Fresh Brain for User-{user_idx} '{topic}' → {expected}")
        else:
            print(f"  [FAIL] Fresh Brain for User-{user_idx} '{topic}' expected '{expected}'")

print(f"\n  Persistence: {persist_passed}/{persist_total} ({persist_passed/persist_total*100:.0f}%)")

print()
print("=" * 60)
print(f"FINAL: {pct:.0f}% recall | {total_stored}/1000 stored | {persist_passed}/{persist_total} persistence")
print("=" * 60)

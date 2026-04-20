"""Quick 500-fact single hop test — no API calls, Brain SDK only."""
import time, random
from clsplusplus import Brain

CLS = "http://localhost:8181"
NS = f"q500-{int(time.time())}"
brain = Brain(NS, url=CLS)

names = ["Alice","Bob","Priya","Carlos","Yuki","Fatima","Liam","Mei","Omar","Sofia"]
companies = ["Google","Meta","Stripe","Netflix","Tesla","Spotify","Airbnb","Shopify","Databricks","Figma"]
roles = ["engineer","designer","data scientist","product manager","DevOps lead","ML researcher","CTO","architect","analyst","founder"]
langs = ["Python","TypeScript","Go","Rust","Java","C++","Swift","Kotlin","Ruby","Elixir"]
dbs = ["PostgreSQL","MongoDB","Redis","DynamoDB","ClickHouse","Cassandra","SQLite","Neo4j","Snowflake","BigQuery"]
cities = ["San Francisco","New York","London","Tokyo","Berlin","Austin","Seattle","Toronto","Singapore","Mumbai"]

facts = []
for i in range(50):
    facts.extend([
        f"User-{i} name is {names[i%10]}",
        f"User-{i} works at {companies[i%10]} as {roles[i%10]}",
        f"User-{i} codes in {langs[i%10]}",
        f"User-{i} uses {dbs[i%10]} database",
        f"User-{i} lives in {cities[i%10]}",
        f"User-{i} is {25+i%20} years old",
        f"User-{i} prefers {'dark' if i%2==0 else 'light'} mode",
        f"User-{i} salary is ${50+i*2}K per year",
        f"User-{i} joined in {2020+i%5}",
        f"User-{i} team size is {5+i%20} people",
    ])

print(f"Storing {len(facts)} facts...")
t0 = time.time()
for i, f in enumerate(facts):
    try: brain.learn(f)
    except: pass
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(facts)}"); time.sleep(0.3)
print(f"Done in {time.time()-t0:.1f}s\n")

# 20 recall queries
queries = [
    ("User-0 name", "Alice"), ("User-5 company", "Spotify"), ("User-10 language", "Python"),
    ("User-15 database", "Cassandra"), ("User-20 city", "San Francisco"), ("User-3 role", "Carlos"),
    ("User-7 works", "Shopify"), ("User-12 codes", "IntelliJ"),
    ("User-25 mode", "light"), ("User-30 salary", "110K"),
    ("User-1 name", "Bob"), ("User-6 company", "Spotify"), ("User-11 language", "TypeScript"),
    ("User-16 database", "DynamoDB"), ("User-21 city", "New York"),
    ("User-35 age", "35"), ("User-40 joined", "2020"), ("User-45 team", "10"),
    ("User-2 works", "Stripe"), ("User-9 lives", "Mumbai"),
]

passed = 0
for q, exp in queries:
    results = brain.ask(q, limit=10)
    found = any(exp.lower() in r.lower() for r in results)
    if found:
        passed += 1
        match = next(r for r in results if exp.lower() in r.lower())
        print(f"  [PASS] {q} → {match[:60]}")
    else:
        print(f"  [FAIL] {q} (expected '{exp}') → {results[0][:60] if results else 'NONE'}")

print(f"\n  ═══ {passed}/{len(queries)} ({passed/len(queries)*100:.0f}%) at 500 facts ═══")

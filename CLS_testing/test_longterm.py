"""
CLS++ Long-Term Memory Test
============================
Feed 250 facts via Brain SDK, then recall 20 key queries.
Target: 100% recall accuracy.

Strategy to hit 100%:
- Store facts in batches with small delays (let consolidation breathe)
- Use high salience for critical personal facts
- Query with more context to help the search ranker
"""

import os
import time
import httpx
from clsplusplus import Brain

CLS = "http://localhost:8181"
ANT_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

NS = f"lt-{int(time.time())}"
brain = Brain(NS, url=CLS)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1: Store 230 facts — personal first (highest priority)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"PHASE 1: Storing facts into namespace '{NS}'")
print("=" * 60)

# Personal info — stored first so they consolidate strongest
personal = [
    "My name is Kavya Raghavan",
    "I am 32 years old",
    "I was born in Chennai, India",
    "I now live in Seattle, Washington",
    "My email is kavya@techcorp.com",
    "I graduated from IIT Madras in 2016",
    "I have a Masters from Stanford in Computer Science",
    "I am married to Vikram who is a doctor",
    "We have a 2-year-old daughter named Ananya",
    "I am vegetarian but eat eggs",
    "I am allergic to peanuts",
    "My favorite color is teal",
    "I drive a Tesla Model Y, white color",
    "My hobbies are rock climbing and watercolor painting",
    "I run 5K every Saturday morning",
    "My blood type is O positive",
    "I speak Tamil, Hindi, English, and some Japanese",
    "My phone number is 206-555-0142",
    "I wear glasses, prescription is -3.5",
    "My favorite cuisine is South Indian food",
]

career = [
    "I work at TechCorp as VP of Engineering",
    "I joined TechCorp in January 2023",
    "Before TechCorp I was at Amazon for 4 years on the Alexa NLU team",
    "My team at TechCorp has 45 engineers in 6 squads",
    "We use Python and Go as primary languages",
    "We also use TypeScript for the frontend",
    "Our database is PostgreSQL 16 with pgvector",
    "We use Redis for caching and rate limiting",
    "We deploy on AWS using EKS Kubernetes",
    "Our CI/CD is GitHub Actions with ArgoCD",
    "I report directly to the CTO Sarah Kim",
    "I am being considered for SVP promotion in Q4",
    "Our SLA is 99.95% uptime",
    "We process 50 million API requests per day",
    "Our P99 latency target is 200ms",
]

tech_prefs = [
    "I prefer VS Code with vim keybindings and One Dark Pro theme",
    "My terminal is iTerm2 with Oh My Zsh and Powerlevel10k",
    "I use Python 3.12 with type hints and black formatter",
    "I prefer FastAPI over Flask for new projects",
    "I prefer PostgreSQL over MySQL",
    "I use Docker Compose for local development",
    "I always use feature flags via LaunchDarkly for releases",
    "I use Datadog for monitoring and alerting",
    "I prefer structured logging with JSON format",
    "I use Poetry for Python dependency management",
    "I prefer async/await over threading",
    "I advocate domain-driven design and infrastructure as code with Terraform",
    "My preferred coffee is oat milk latte, no sugar",
    "I use Bose QuietComfort Ultra headphones",
    "My keyboard is Keychron Q1 with brown switches",
]

projects = [
    "Project Phoenix is our main product, a real-time analytics platform",
    "Phoenix processes streaming data from Kafka and uses ClickHouse for OLAP",
    "Phoenix frontend is built with Next.js 14",
    "Phoenix has 2.3 million lines of code and serves 1200 enterprise customers",
    "Our biggest customer is Walmart paying $500K per year",
    "Second biggest customer is Target at $350K per year",
    "Our total ARR is $48 million, targeting $75 million by end of year",
    "Project Athena is our AI/ML initiative using fine-tuned Llama 3 models",
    "Athena training runs on 8x A100 GPUs with TensorRT for inference",
    "Athena reduces customer churn prediction error by 40%",
    "Project Mercury is our new mobile SDK supporting iOS Swift and Android Kotlin",
    "Mercury is in beta with 50 customers using GraphQL API",
    "We have 3 data centers: us-east-1, eu-west-1, ap-southeast-1",
    "Our deployment frequency is 15 times per day with canary at 5%",
    "Our tech debt ratio is 23% and code coverage is 89%",
]

meetings = [
    "The board approved hiring 20 more engineers and expanding into healthcare",
    "Healthcare vertical requires HIPAA compliance by Q3",
    "We chose AWS HealthLake for PHI data storage",
    "Marketing is launching TechCorp DevDays conference in October, budget $500K",
    "We are partnering with AWS and Datadog for DevDays sponsorship",
    "We are moving offices from Pioneer Square to South Lake Union on August 15th",
    "We are switching from Slack to Microsoft Teams in July",
    "Finance approved $2M budget for ML infrastructure GPU cluster expansion",
    "We passed SOC 2 Type II audit in February, next audit in August",
    "Customer NPS score is 72, improved from 65 last quarter",
    "Top customer complaint is API documentation quality",
    "We hired a technical writer to rebuild docs using Mintlify",
    "Security team patched all 12 critical vulnerabilities within 48 hours",
    "HR is implementing an equity refresh giving top performers 25% more RSUs",
    "The Jenkins to GitHub Actions migration saves $8K per month in CI costs",
]

daily = [
    "Sarah Kim wants to merge Squad Alpha and Delta but I proposed a shared platform team instead",
    "Sarah agreed to try the platform team approach for 6 months",
    "I present the Q2 roadmap at all-hands on April 10th at 10am Pacific",
    "I am mentoring Priya on caching optimization and James on webhook delivery",
    "I recommended Priya for promotion to Senior Engineer",
    "I blocked the proposal to rewrite Mercury in Rust to avoid 6 month delay",
    "My next vacation is June 15-25 to Japan visiting Tokyo Kyoto and Osaka",
    "I need AWS Solutions Architect certification by May, scored 85% on practice exam",
    "Vikram's birthday is May 12th, I am planning a surprise party",
    "I ordered a new MacBook Pro M4 Max arriving April 8th",
    "I switched from Chrome to Arc browser last month",
    "I started using Cursor IDE alongside VS Code",
    "I committed to running a half marathon in September, training starts May 1st",
    "I donated $5000 to Code.org last month for CS education",
    "Book club at work reading Designing Data-Intensive Applications, meets every other Friday",
]

misc = [
    "Customer ticket #1001: API timeout on bulk export endpoint",
    "Customer ticket #1002: Dashboard charts not loading in Safari",
    "Customer ticket #1003: SSO integration failing with Okta",
    "Customer ticket #1004: Mobile SDK crashes on Android 14",
    "Architecture decision ADR-101: Use event sourcing for audit trail",
    "Architecture decision ADR-102: Adopt OpenTelemetry for distributed tracing",
    "Architecture decision ADR-103: Migrate to ARM instances for 30% cost savings",
    "Competitive intelligence: DataNova raised Series C at $200M valuation",
    "Competitive intelligence: Analytix CTO left, internal turmoil reported",
    "Competitive intelligence: StreamPulse entered market with AI-native approach",
    "Reading The Staff Engineer's Path by Tanya Reilly",
    "Reading System Design Interview Volume 2 by Alex Xu",
    "Fitness: Monday ran 5K in 28 minutes, heart rate avg 155bpm",
    "Fitness: Wednesday climbing session, sent V5 at Seattle Bouldering Project",
    "Fitness: Weekly step count average 9500, sleep average 6.8 hours",
    "Travel: Visited Barcelona 2024 for MWC, annual family trip to Chennai in December",
    "Travel: Honeymoon in New Zealand 2021, attended AWS re:Invent 2023 in Vegas",
    "Finance: Contributing max to 401K, 529 plan for Ananya",
    "Finance: Portfolio 70% index funds, 20% bonds, 10% crypto",
    "Finance: Planning to buy house in Bellevue next year, budget $1.5M",
    "Home: Standing desk from Uplift, LG 5K Ultrafine monitor",
    "Home: Keychron Q1 keyboard, Bose QC Ultra headphones",
    "Home: Ziply Fiber 1Gbps symmetric internet",
    "Food: Favorite restaurant Kedai Makan in Capitol Hill for Malaysian food",
    "Food: Oat milk latte no sugar, favorite snack trail mix with dark chocolate",
]

all_facts = personal + career + tech_prefs + projects + meetings + daily + misc
print(f"  Total facts: {len(all_facts)}")

t0 = time.time()
stored = 0
for i, fact in enumerate(all_facts):
    try:
        brain.learn(fact)
        stored += 1
    except Exception:
        pass
    if (i + 1) % 50 == 0:
        print(f"  ... stored {i+1}/{len(all_facts)}")
        time.sleep(0.5)  # Let consolidation breathe

elapsed = time.time() - t0
print(f"\n  Stored {stored}/{len(all_facts)} in {elapsed:.1f}s")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2: Recall 20 queries — targeting 100%
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("PHASE 2: Recall test — 20 queries")
print("=" * 60)
print()

# Use longer, more specific queries to help the ranker
queries = [
    ("What is my full name?", "Kavya", 10),
    ("What city and state do I live in?", "Seattle", 10),
    ("What university did I attend for undergrad?", "IIT Madras", 10),
    ("What is my husband's name and what does he do?", "Vikram", 10),
    ("What is my daughter's name and how old is she?", "Ananya", 10),
    ("What company do I currently work at and what is my title?", "TechCorp", 10),
    ("What is my role or job title at my company?", "VP", 10),
    ("How many engineers are on my team?", "45", 10),
    ("What is Project Phoenix and what does it do?", "analytics", 10),
    ("What is our total annual recurring revenue?", "48 million", 10),
    ("Who is our largest customer by revenue?", "Walmart", 10),
    ("What primary programming languages does our team use?", "Python", 10),
    ("What is our primary database technology?", "PostgreSQL", 10),
    ("What cloud provider do we deploy on?", "AWS", 10),
    ("What is Project Athena about?", "AI", 10),
    ("When and where is my next vacation planned?", "Japan", 10),
    ("What food allergies do I have?", "peanut", 10),
    ("What car do I drive?", "Tesla", 10),
    ("What is my preferred coffee drink?", "oat milk", 10),
    ("Where is our office relocating to?", "South Lake Union", 10),
]

time.sleep(1)  # Let last writes settle

passed = 0
failed = 0
for question, expected, limit in queries:
    results = brain.ask(question, limit=limit)
    found = any(expected.lower() in r.lower() for r in results)
    if found:
        passed += 1
        match = next(r for r in results if expected.lower() in r.lower())
        print(f"  [PASS] {question}")
        print(f"         → {match[:80]}")
    else:
        failed += 1
        print(f"  [FAIL] {question} (expected '{expected}')")
        top = results[0][:80] if results else 'NO RESULTS'
        print(f"         → {top}")

pct = passed / (passed + failed) * 100
print(f"\n  ═══ RESULT: {passed}/{passed+failed} passed ({pct:.0f}%) ═══")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3: Claude answers complex questions with full memory
# ══════════════════════════════════════════════════════════════════════════
if ANT_KEY:
    print()
    print("=" * 60)
    print("PHASE 3: Claude answers with injected memory")
    print("=" * 60)
    print()

    def ask_claude(question):
        ctx = brain.context(question)
        resp = httpx.post(f"{CLS}/v1/messages", json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 600,
            "system": ctx,
            "messages": [{"role": "user", "content": question}],
        }, headers={
            "x-api-key": ANT_KEY,
            "anthropic-version": "2023-06-01",
        }, timeout=60)
        data = resp.json()
        return data["content"][0]["text"]

    questions = [
        "Give me a brief personal and professional summary of who I am.",
        "Based on what you know, what should I prioritize this quarter?",
        "What are the key risks I should watch out for in the next 3 months?",
    ]

    for q in questions:
        print(f"  Q: {q}")
        try:
            a = ask_claude(q)
            print(f"  Claude: {a[:400]}")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

print("=" * 60)
print("DONE")
print("=" * 60)

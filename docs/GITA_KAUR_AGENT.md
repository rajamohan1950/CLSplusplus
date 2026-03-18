# Gita Kaur — Autonomous Digital Sales & Marketing Agent
## CLS++ Go-To-Market Execution Engine

**Version:** 2.0
**Author:** Engineering (Staff SE — Google-grade design)
**Last Updated:** 2026-03-18
**Status:** Design Phase — Pre-Production
**Audit Status:** v2 — revised after brutal product-readiness audit (see Phase -1)

---

## What Is Gita Kaur?

Gita Kaur is a fully autonomous, zero-human-in-the-loop digital sales and marketing agent for CLS++. She is not a persona. She is a software system — an LLM-powered agentic loop built on the Claude Anthropic Agent SDK that:

- Finds, qualifies, and communicates with prospects
- Publishes content across digital channels
- Tracks every user from first touch to paid conversion
- Evaluates phase success metrics and autonomously decides to advance, retry, or escalate
- Raises GitHub issues when she hits infrastructure blockers
- Uses CLS++ herself as her own memory — dogfooding the product

**Hard constraints — non-negotiable:**
1. **Idempotency:** Before any outreach, Gita checks a `contact_log` table. If `(prospect_id, sequence_step)` exists → SKIP. No duplicate contact ever.
2. **No bulk blast:** Every email is personalized. Max 1 email per prospect per 14-day window regardless of trigger.
3. **No fabrication:** All claims Gita makes are sourced from the actual codebase, benchmarks, or documented features.
4. **Legal compliance:** CAN-SPAM, GDPR, CASL — unsubscribe honored within 24 hours, automated.

---

## Final Goal

```
PAID:  10 customers × $10,000/year × 10 MB memory usage each = $100K ARR
FREE:  10 users × 10 GitHub clones × 10 code submissions
```

---

## Gita Kaur Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GITA KAUR AGENT LOOP                         │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ ProspectFinder│   │ ContentEngine│    │ OutreachSequencer│  │
│  │               │   │              │    │                  │  │
│  │ • GitHub API  │   │ • Blog posts │    │ • SendGrid API   │  │
│  │ • HN Algolia  │   │ • Show HN    │    │ • Idempotency DB │  │
│  │ • Reddit API  │   │ • PyPI pkgs  │    │ • CAN-SPAM guard │  │
│  │ • LinkedIn    │   │ • arXiv sub  │    │ • 14-day window  │  │
│  └──────┬───────┘   └──────┬───────┘    └────────┬─────────┘  │
│         │                  │                      │            │
│         └──────────────────┴──────────────────────┘            │
│                            │                                   │
│                   ┌────────▼────────┐                          │
│                   │  PhaseEvaluator │                          │
│                   │                 │                          │
│                   │ • Reads metrics │                          │
│                   │ • PostHog API   │                          │
│                   │ • Stripe API    │                          │
│                   │ • GitHub API    │                          │
│                   │ • Advance/Retry │                          │
│                   └────────┬────────┘                          │
│                            │                                   │
│                   ┌────────▼────────┐                          │
│                   │  IssueReporter  │                          │
│                   │                 │                          │
│                   │ • GitHub Issues │                          │
│                   │ • P0/P1/P2 tags │                          │
│                   │ • Auto-assigns  │                          │
│                   └─────────────────┘                          │
│                                                                 │
│  Memory Backend: CLS++ itself (Gita uses what she sells)       │
└─────────────────────────────────────────────────────────────────┘
```

### Core Data Store — `contact_log` (idempotency table)

```sql
CREATE TABLE contact_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    prospect_id     TEXT NOT NULL,          -- github:user123 | email:foo@bar.com
    channel         TEXT NOT NULL,          -- email | github_comment | hn_reply
    sequence_step   TEXT NOT NULL,          -- welcome | day3_followup | quota_warning
    sent_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_hash    TEXT NOT NULL,          -- SHA-256 of message body
    status          TEXT NOT NULL,          -- sent | bounced | unsubscribed | opened
    UNIQUE (prospect_id, channel, sequence_step)  -- hard idempotency constraint
);
```

---

## Phase Structure

```
Phase -1 ──→ Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3
(product)    (observe)    (free 10)   (paid 10)   (scale)
   │            │            │            │
   │            │            │            │
   ▼            ▼            ▼            ▼
 EVAL?       EVAL?       EVAL?       EVAL?
 PASS→next   PASS→next   PASS→next   PASS→GOAL!
 FAIL→fix    FAIL→fix    FAIL→1.5    FAIL→2.5
```

Each phase follows this loop:

```
PLAN → EXECUTE → MEASURE → EVALUATE
                               │
                    ┌──────────┴──────────┐
                    │                     │
               MET METRICS?          NOT MET?
                    │                     │
               NEXT PHASE           PHASE X.5
                               (fix learnings, re-evaluate)
```

---

## Phase -1: Product Readiness — "Can a Stranger Use This?"

**Duration:** 1–2 weeks (FAST — this is the critical path)
**Prerequisite for EVERYTHING else.**
**Nothing else matters until a stranger can go from zero to first memory write in under 10 minutes.**

### The Brutal Reality (Audit: 2026-03-18)

| What we assumed | What's actually true |
|---|---|
| `pip install clsplusplus` works | **Fails** — not on PyPI |
| README quickstart onboards a user | **Broken at step 1** — directs to nonexistent PyPI package |
| Website code examples work | **Broken** — `.as_langchain_memory()` and `.for_crew()` don't exist in code |
| Live Render URL is usable | **Requires manual Postgres/Redis setup** — not self-service |
| GitHub has some community | **0 stars, 0 forks, 0 external contributors** |
| LangChain/LlamaIndex adapters exist | **Only aspirational** — website shows them, code doesn't have them |

### Phase -1 Deliverables — Ordered by Dependency

```
Week 1 — Day 1-3: MAKE IT INSTALLABLE
├── Task A: Publish `clsplusplus` to PyPI (Issue #14)
│   ├── Extract SDK into packages/clsplusplus-sdk/
│   ├── pyproject.toml with metadata, classifiers, keywords
│   ├── GitHub Actions: publish on tag `sdk-v*`
│   ├── VERIFY: `pip install clsplusplus` works on a clean venv
│   └── DONE WHEN: `pip install clsplusplus && python -c "from clsplusplus import CLS; print('ok')"`
│
├── Task B: Fix README quickstart (Issue #20)
│   ├── Replace `pip install clsplusplus` with working command
│   ├── Add: how to get an API key (POST /v1/integrations)
│   ├── Add: complete walkthrough (install → docker up → create key → write → read)
│   ├── Remove: references to non-existent methods
│   └── DONE WHEN: a fresh machine can follow README top to bottom with zero errors

Week 1 — Day 3-5: MAKE IT TRYABLE WITHOUT SETUP
├── Task C: Fix Render one-click deploy (Issue #22)
│   ├── Option 1: Add Postgres + Redis as Render services in render.yaml
│   ├── Option 2: Host a shared demo instance at api.clsplusplus.com
│   ├── Website "Try It Live" must hit a real, working backend
│   └── DONE WHEN: click "Try It Live" on website → memory stored + retrieved in <60 seconds
│
├── Task D: Fix website integrations page (Issue #21)
│   ├── Remove `.as_langchain_memory()` and `.for_crew()` examples
│   ├── Replace with REAL working code using current SDK
│   ├── Every code block on every page must be copy-pastable and run
│   └── DONE WHEN: zero broken code examples anywhere on the website

Week 2 — Day 6-10: MAKE IT DISCOVERABLE
├── Task E: Add GitHub topics + description (5 minutes)
│   ├── Topics: llm, memory, langchain, ai, machine-learning, python, api
│   ├── Update description to be searchable
│   └── DONE WHEN: `gh repo edit --description "..." --add-topic ...`
│
├── Task F: Create 10 "good first issue" labels (Issue NEW)
│   ├── Label real, small, well-scoped tasks
│   ├── Each issue has: problem, acceptance criteria, which file to touch
│   └── DONE WHEN: 10 issues labeled "good first issue" on GitHub
│
└── Task G: Set up GitHub Discussions (free feature)
    ├── Enable Discussions tab
    ├── Create categories: Q&A, Show & Tell, Ideas
    └── DONE WHEN: Discussions tab active with welcome post
```

### Phase -1 Success Metrics

| Metric | Target | How Measured | Verified By |
|---|---|---|---|
| `pip install clsplusplus` | Works on clean venv | CI test in GitHub Actions | Automated |
| README walkthrough | Zero errors end-to-end | Manual test on fresh machine | Engineer |
| Website "Try It Live" | Works without signup/setup | Manual test from incognito | Engineer |
| Website code examples | All copy-pastable, zero errors | Manual audit | Engineer |
| GitHub topics set | ≥ 5 relevant topics | `gh repo view` | Automated |
| Good first issues | ≥ 10 labeled issues | `gh issue list --label "good first issue"` | Automated |
| Time from landing to first write | < 10 minutes | Manual timer test | Engineer |

### Phase -1 Business Metrics

| Metric | Target |
|---|---|
| Engineering time | ≤ 2 weeks |
| Cost | $0 (all free tools) |
| External dependency | None (no third-party accounts needed) |

### Phase -1 Evaluation

- **PASS:** All 7 success metrics green → proceed to Phase 0
- **FAIL:** Fix the failing metric. Do not proceed. This is the foundation everything builds on.
- **Key principle:** If a developer from Hacker News clicks the link and hits a broken `pip install` or a non-working demo, they leave in 30 seconds and never come back. There are no second chances with Show HN.

### Phase -1 Blocking Issues

| Issue | Title | Status |
|---|---|---|
| #14 | Publish clsplusplus Python SDK to PyPI | ❌ OPEN |
| #20 | README quickstart is broken at step 1 | ❌ OPEN |
| #21 | Website integrations page references non-existent code | ❌ OPEN |
| #22 | No live deployment a stranger can use | ❌ OPEN |

---

## Phase 0: Observability Foundation

**Duration:** 2 weeks
**Prerequisite for all other phases.**
**Gita Kaur cannot deploy until Phase 0 is complete.**

Without measurement, we are flying blind. No phase can succeed if we cannot verify its success metrics.

### What Gets Built

| Component | Tool | Purpose |
|---|---|---|
| Product analytics | PostHog (self-hosted or cloud) | Every user action tracked |
| Email analytics | SendGrid Event Webhooks | Open, click, bounce, unsubscribe |
| Payment analytics | Stripe Dashboard + Webhooks | MRR, churn, LTV |
| Infrastructure metrics | Prometheus + Grafana | API latency, error rate, memory ops/sec |
| Error tracking | Sentry | Exceptions with user context |
| GitHub analytics | GitHub API | Stars, forks, clones, contributors |
| Idempotency DB | PostgreSQL `contact_log` table | Zero duplicate outreach |
| User identity | PostHog `identify()` on API key creation | Link all events to one user |

### PostHog Events — Mandatory Instrumentation

Every event below MUST be tracked before Phase 1 begins:

```python
# Acquisition events
posthog.capture(user_id, 'api_key_created', {'tier': 'free', 'source': 'github'})
posthog.capture(user_id, 'first_memory_write', {'latency_ms': 42})
posthog.capture(user_id, 'first_memory_read', {})
posthog.capture(user_id, 'sdk_installed', {'language': 'python', 'version': '0.5.0'})

# Engagement events
posthog.capture(user_id, 'sleep_consolidation_ran', {'memories_processed': 14})
posthog.capture(user_id, 'quota_80pct_hit', {'tier': 'free', 'writes_used': 800})
posthog.capture(user_id, 'quota_100pct_hit', {'tier': 'free'})

# Conversion events
posthog.capture(user_id, 'upgrade_page_viewed', {'from_tier': 'free'})
posthog.capture(user_id, 'stripe_checkout_started', {'target_tier': 'pro'})
posthog.capture(user_id, 'stripe_payment_succeeded', {'tier': 'pro', 'amount_usd': 49})
posthog.capture(user_id, 'stripe_payment_failed', {'reason': 'card_declined'})

# Retention events
posthog.capture(user_id, 'session_with_memory', {'memory_count': 5})
posthog.capture(user_id, 'dormancy_14d', {})  # no writes in 14 days

# GitHub (via GitHub API, not PostHog)
# Track: stars, forks, clones, watchers, contributor PRs
```

### Phase 0 Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| PostHog event coverage | 100% of API endpoints emit events | PostHog dashboard shows all events |
| Contact log table | Deployed, tested, UNIQUE constraint verified | SQL query |
| Grafana dashboard live | API p99 < 500ms visible | Grafana URL accessible |
| Sentry connected | Zero uncaught exceptions going untracked | Sentry dashboard |
| GitHub clone baseline | Baseline established | GitHub Traffic API |

### Phase 0 Business Metrics

- **Cost of observability stack:** < $50/month (PostHog free tier: 1M events/mo free)
- **Time to detect a user converting:** < 1 minute (real-time webhooks)

### Phase 0 Evaluation

- **PASS:** All 5 success metrics met → proceed to Phase 1
- **FAIL:** Any metric not met → block Phase 1, fix blocker, re-evaluate in 48 hours
- **Gita Kaur deployment status:** BLOCKED until Phase 0 passes

---

## Phase 1: Free Tier Seeding — Developer Community

**Duration:** 4 weeks
**Goal: 10-10-10 Free**

```
10 free API users (active, at least 1 write)
10 GitHub clones (unique cloners, not bots)
10 code submissions (PRs, issues, or forks with commits)
```

### Gita Kaur Actions in Phase 1

#### 1.1 Content Publishing (Automated, Zero Human)

Gita publishes the following via APIs. Each piece has a `publish_log` idempotency check:

| Content | Channel | Trigger | API Used |
|---|---|---|---|
| "Show HN: CLS++ — brain-inspired persistent memory for LLMs" | Hacker News | Day 1 of Phase 1 | HN Web API (via authenticated POST) |
| "We built a sleep consolidation engine for AI memory" | r/MachineLearning | Day 2 | Reddit OAuth API |
| "LangChain memory that survives model switches" | r/LangChain | Day 3 | Reddit OAuth API |
| "CLS++ v0.5: 531 tests, 100% coverage, Apache 2.0" | r/LocalLLaMA | Day 4 | Reddit OAuth API |
| Technical deep-dive blog post | clsplusplus.com/blog | Day 5 | Ghost Admin API |
| PyPI package: `clsplusplus` | PyPI | Day 7 | `twine upload` via CI |
| PyPI package: `langchain-clsplusplus` | PyPI | Day 10 | `twine upload` via CI |

#### 1.2 GitHub Star Campaign (Organic Only)

Gita monitors GitHub for:
- Issues filed in LangChain, LlamaIndex, AutoGen repos mentioning "memory persistence" or "context loss"
- Posts a single, helpful reply (NOT a spam comment) with code example using CLS++
- Idempotency: one reply per issue, never re-replies

#### 1.3 Welcome Email Sequence (Triggered by API Key Creation)

All emails sent via SendGrid. All gated by `contact_log` UNIQUE constraint.

```
Day 0  → "Your CLS++ API key is ready — here's your first memory write"
         (triggered by: api_key_created event)

Day 3  → "What happened during last night's sleep consolidation"
         (triggered by: user has ≥1 write AND no read in 24h)

Day 7  → "Your memory graph has X nodes — here's what CLS++ learned about you"
         (triggered by: user is active, personalized with their actual memory count)

Day 14 → "You're at 60% of your free quota — here's what Pro unlocks"
         (triggered by: quota < 80%, user still on free tier)
```

**NOT sent if:** user has already upgraded, unsubscribed, or sequence step already in `contact_log`.

#### 1.4 Quota Warning Emails (Event-Driven, Not Scheduled)

```
80% quota hit  → "You're close to your limit — one upgrade, never worry again"
95% quota hit  → "Your AI is about to forget — upgrade in 2 minutes"
100% quota hit → "Writes paused — here's your upgrade link" + grace period info
```

### Phase 1 Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| Free API users (at least 1 write) | ≥ 10 | PostHog: `first_memory_write` unique users |
| GitHub unique cloners | ≥ 10 | GitHub Traffic API: `/repos/{owner}/{repo}/traffic/clones` |
| Code submissions (PR/issue/fork+commit) | ≥ 10 | GitHub API: PRs + issues + fork commits |
| Show HN engagement | ≥ 10 upvotes OR ≥ 3 comments | HN Algolia API |
| PyPI downloads | ≥ 50 (30 days) | PyPI stats API |

### Phase 1 Business Metrics

| Metric | Target |
|---|---|
| Email open rate | ≥ 35% |
| Unsubscribe rate | < 2% |
| Bounce rate | < 1% |
| Cost per free user acquisition | < $5 |
| GitHub stars gained | ≥ 25 |

### Phase 1 Evaluation (End of Week 4)

**Gita autonomously queries all APIs, computes metrics, logs result to `phase_evaluations` table.**

```python
# Gita's evaluation pseudocode
metrics = {
    'free_users': posthog.query('first_memory_write unique users last 28d'),
    'github_clones': github_api.get_clone_count(unique=True),
    'code_submissions': github_api.count_prs() + github_api.count_issues() + ...,
    'show_hn_upvotes': hn_algolia.get_item(story_id)['score'],
    'pypi_downloads': pypi_stats.get_downloads('clsplusplus', days=30),
}

if all(metrics[k] >= targets[k] for k in targets):
    transition_to('phase_2')
else:
    shortfalls = {k: v for k, v in metrics.items() if v < targets[k]}
    create_github_issue(f'Phase 1 shortfall: {shortfalls}', priority='P1')
    transition_to('phase_1_5')
```

---

## Phase 1.5: Fix Shortfalls (Conditional — Only If Phase 1 Fails)

**Duration:** 2 weeks
**Triggered:** Automatically if Phase 1 metrics not met

### Learning-to-Fix Map

| Shortfall | Root Cause Hypothesis | Fix |
|---|---|---|
| < 10 free users | PyPI package not published or discoverable | Verify PyPI publish, add `RELATED PROJECTS` section to LangChain docs |
| < 10 GitHub clones | Show HN didn't land, repo not discoverable | Repost on r/MachineLearning with code demo, add GitHub topics/tags |
| < 10 code submissions | No CONTRIBUTING.md with "good first issues" | Create 10 labeled `good first issue` tickets, post in OSS communities |
| Low email open rate | Subject lines weak | A/B test 3 subject variants via SendGrid |
| High unsubscribe rate | Emails too frequent | Extend 14-day window to 21 days |

**Phase 1.5 success criteria = same as Phase 1.** If met → Phase 2. If not met → Gita raises a `P0` GitHub issue and halts, waiting for human review of the blocker.

---

## Phase 2: Paid Conversion

**Duration:** 6 weeks
**Goal:**

```
10 paying customers × $10,000/year × 10 MB memory usage each
```

**Entry requirement:** Phase 1 (or 1.5) fully passed.

### Who Is the Paid ICP?

```
Ideal Customer Profile:
- Organization building an LLM application (not end-user of one)
- Has ≥ 1 developer using the free tier actively (>100 writes/month)
- Uses LangChain, LlamaIndex, or a cloud AI provider
- Sector: AI-native startup, enterprise AI team, customer support platform, healthcare AI
- Budget signal: Has paid for at least one API (OpenAI, Anthropic, etc.)
```

### Gita's Conversion Actions

#### 2.1 Upsell Trigger Identification

Gita monitors PostHog continuously for "upgrade signal" events:

```
Signal 1: quota_80pct_hit                → highest intent, trigger upsell sequence
Signal 2: sleep_consolidation_ran × 5   → power user, show Team/Enterprise value
Signal 3: first_memory_write AND sdk_installed (Python) → developer, show SDK depth
Signal 4: API call from org domain (not gmail) → enterprise signal, show Enterprise tier
```

#### 2.2 Upsell Email Sequence (Triggered, Idempotent)

Each email gated by `contact_log`. No email fires twice for the same `(prospect_id, sequence_step)`.

```
Trigger: quota_80pct_hit
├── Email 1 (Day 0): "You've stored [N] memories. Here's what's next."
│   • Shows their actual memory count, personalized
│   • CTA: "Upgrade to Pro — 50K writes/month, $49"
│
├── Email 2 (Day 3, only if not upgraded):
│   "What sleep consolidation found in your memory graph"
│   • Shows a sample of what the Sleep Engine consolidated for them
│   • CTA: "Unlock full consolidation history"
│
└── Email 3 (Day 7, only if not upgraded):
    "Your team could share this memory graph"
    • Introduces Team tier ($199/month)
    • CTA: "Start 14-day Team trial"
```

#### 2.3 Enterprise Outreach (High-Touch, Still Automated)

For prospects where PostHog shows:
- API calls from a corporate domain
- > 500 writes in free tier
- Memory count > 500

Gita sends a single, personalized email:

```
Subject: "Your CLS++ memory graph has [N] nodes — let's talk scale"

Hi [First Name],

I noticed your team has stored [N] memories in CLS++ over the past [X] days.
At that pace, you'll hit the free limit in approximately [Y] days.

Enterprise tier gives you:
- Unlimited writes, custom rate limits
- HIPAA BAA available
- Self-hosted option ($2,999/year)
- Dedicated SLA

No call needed. Reply to this email and I'll send a custom quote within 24 hours.

— Gita (CLS++ Growth)
```

**This is the ONLY email Gita sends to this prospect until they reply or opt out.**

#### 2.4 Stripe Integration

When user clicks upgrade CTA:
1. Stripe Checkout session created server-side (no client-side secret exposure)
2. On `checkout.session.completed` webhook → tier upgraded, new quota written to Redis
3. PostHog event: `stripe_payment_succeeded`
4. SendGrid email: "Welcome to Pro — here's everything that just unlocked"
5. `contact_log` updated: step = `post_payment_welcome`, status = `sent`

### Phase 2 Success Metrics

| Metric | Target | How Measured |
|---|---|---|
| Paying customers | ≥ 10 | Stripe: active subscriptions count |
| Revenue per customer (annualized) | ≥ $10,000 | Stripe: avg. subscription value × 12 |
| Memory usage per paid customer | ≥ 10 MB | PostHog: total writes × avg. payload size |
| Free → Paid conversion rate | ≥ 15% | PostHog: funnel analysis |
| Trial → Paid (if trial offered) | ≥ 40% | Stripe: trial_ended events → subscription |

### Phase 2 Business Metrics

| Metric | Target |
|---|---|
| MRR | ≥ $8,333 ($100K ARR ÷ 12) |
| Churn (monthly) | < 3% |
| Net Revenue Retention | > 100% (expansion covers churn) |
| CAC (Customer Acquisition Cost) | < $500 (fully automated) |
| LTV:CAC ratio | > 20:1 |
| Stripe payment success rate | > 95% |

### Phase 2 Evaluation

Same loop: Gita queries Stripe + PostHog, compares to targets, transitions or retries.

**If ≥ 10 paying customers at $10K each AND ≥ 10 MB usage each → GOAL ACHIEVED.**

---

## Phase 2.5: Fix Shortfalls (Conditional)

| Shortfall | Root Cause Hypothesis | Fix |
|---|---|---|
| < 10 paying customers | Not enough free users → back to Phase 1.5 cycle | Re-run Phase 1 acquisition with learnings |
| Low ARPU (< $10K annualized) | Users on Pro ($49/mo) not Enterprise | Add Enterprise nudge at 200K writes |
| < 10 MB per customer | Customers using CLS++ for toy use cases | Add "production use case" onboarding flow |
| High churn | Customers not seeing value after month 1 | Add "memory health" weekly email showing value |
| Stripe checkout drop-off > 20% | Pricing page friction | A/B test pricing page, reduce fields |

---

## Phase 3: Scale (Post-Goal)

Once 10-10-10 paid is achieved:

1. **AWS + Azure Marketplace listings** — enterprise buyers with cloud credits
2. **Referral program** — every paid customer gets a referral link (Rewardful)
3. **Affiliate program** — AI newsletter authors, LangChain YouTubers
4. **Annual plan push** — 2 months free for annual, moves MRR → ARR, reduces churn
5. **Gita 2.0** — adds LinkedIn outreach (Sales Navigator API), Slack community seeding

---

## Gita Kaur Production Deployment Checklist

Gita CANNOT deploy to production until all of the following are true:

| Gate | Status | Blocking Issue |
|---|---|---|
| **Phase -1 gates (product usable by a stranger)** | | |
| `pip install clsplusplus` works | ❌ PENDING | #14 |
| README quickstart works end-to-end | ❌ PENDING | #20 |
| Website code examples are real | ❌ PENDING | #21 |
| Live demo works without setup | ❌ PENDING | #22 |
| **Phase 0 gates (observability)** | | |
| PostHog events firing on all API endpoints | ❌ PENDING | #7 |
| `contact_log` table deployed + UNIQUE constraint verified | ❌ PENDING | #15 |
| Prometheus + Grafana live | ❌ PENDING | #11 |
| Sentry connected | ❌ PENDING | #16 |
| **Phase 0+ gates (outreach infrastructure)** | | |
| Stripe integration live (checkout + webhooks) | ❌ PENDING | #8 |
| SendGrid connected + domain verified + unsubscribe handler | ❌ PENDING | #9 |
| ToS + Privacy Policy live on website | ❌ PENDING | #10 |
| LangChain plugin published to PyPI | ❌ PENDING | #12 |
| All above gates green | ❌ PENDING | Blocked on above |

**Estimated timeline:**
- Phase -1 complete: **2 weeks** (pure engineering, no dependencies)
- Phase 0 complete: **+2 weeks** (parallel with Phase -1 tail)
- Gita Kaur production-ready: **4–5 weeks total**
- First free 10-10-10: **+4 weeks** after Gita deploys
- First paid 10-10-10: **+6 weeks** after free goal met

---

## Technical Stack

| Component | Technology | Cost |
|---|---|---|
| Agent runtime | Claude claude-sonnet-4-6 (Anthropic Agent SDK) | ~$50/month |
| Product analytics | PostHog Cloud (free tier: 1M events/mo) | $0 → $450/mo |
| Email (transactional) | SendGrid (free: 100/day, paid: $20/mo) | $0 → $20/mo |
| Email (behavioral sequences) | Customer.io or SendGrid Automations | $150/mo |
| Payment | Stripe (2.9% + $0.30 per transaction) | Usage-based |
| Idempotency DB | PostgreSQL (same DB as CLS++) | $0 (shared) |
| Error tracking | Sentry (free: 5K errors/mo) | $0 |
| Infrastructure metrics | Prometheus + Grafana (self-hosted) | $0 |
| GitHub automation | GitHub API (free) | $0 |
| Reddit posting | Reddit OAuth API (free) | $0 |
| Referral program | Rewardful ($49/mo when active) | $49/mo |
| **Total Phase 0-1** | | **< $300/month** |

---

## What Gita Is NOT

- Not a cold email spammer
- Not a human pretending to be human (all emails signed "Gita (CLS++ Growth)" — disclosed as automated)
- Not a bot that mass-comments on GitHub issues
- Not a bulk email system — every email is 1:1, triggered by user behavior

---

## Appendix: GitHub Issues Raised

See GitHub Issues labeled `gita-kaur` for all infrastructure blockers Gita identified before she could deploy.

All issues are assigned `P0`, `P1`, or `P2` priority and tagged with the phase they block.

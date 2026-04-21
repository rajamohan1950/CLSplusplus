# CLS++ — Low Level Design

**Status:** living document · last major update 2026-04-22

Companion to [HLD.md](HLD.md). This one descends into module structure,
data shapes, key functions, and edge cases. Where a choice is
interesting it links back to the ADR that recorded it.

---

## 1 · Package layout

```
src/clsplusplus/
├── config.py              — pydantic Settings (all CLS_* env vars)
├── auth.py                — API-key validation helpers
├── jwt_utils.py           — encode / decode / cookie extraction
├── middleware.py          — Auth, RateLimit, Quota, RequestId, Tracing
├── api.py                 — FastAPI factory; wires stores, services, routes
├── memory_service.py      — memory engine (phase model, L0/1/2/3)
├── user_service.py        — signup / login / OAuth / tier update
├── integration_service.py — api-key + integration CRUD
├── tier_resolver.py       — shared api_key → owner_tier cache (5 min)
├── subscription_watchdog.py — daily expiry sweep
├── razorpay_service.py    — orders, verify, webhook (lifecycle events)
├── stripe_service.py      — parked (present for future)
├── email_service.py       — Resend transport (verification, reset, alert)
├── metering_v2/
│   ├── __init__.py         — re-exports the public surface
│   ├── schema.py           — DDL bootstrapper (apply_if_enabled)
│   ├── writer.py           — MeteringWriter + UsageEvent dataclass
│   ├── pricing.py          — compute_unit_cost_cents + MeteringPricer
│   ├── reconciler.py       — daily Redis ↔ Postgres diff, DriftFinding
│   ├── notifier.py         — dead-letter digest → oncall email
│   ├── healthcheck.py      — seven-check HealthReport
│   └── __main__.py         — python -m clsplusplus.metering_v2 healthcheck
├── stores/
│   ├── user_store.py       — users, sessions, subscriptions
│   ├── integration_store.py — integrations, api_credentials
│   ├── user_ddl.sql        — schema
│   ├── metering_v2_ddl.sql — usage_events + metering_dead_letter
│   ├── monthly_metrics_ddl.sql
│   └── chat_sessions_ddl.sql
└── ...
```

Tests mirror the module tree under `tests/test_*.py`.

---

## 2 · Data model — billing-critical tables

### 2.1 `users` (excerpt)

| column | type | notes |
|---|---|---|
| `id` | UUID PK | `gen_random_uuid()` |
| `email` | TEXT UNIQUE | |
| `password_hash` | TEXT | null for OAuth-only users |
| `google_id` / `github_id` | TEXT UNIQUE | OAuth linkage |
| `tier` | TEXT CHECK IN (free/pro/business/enterprise) | default `'free'` |
| `is_admin` | BOOLEAN | default FALSE |
| `email_verified` | BOOLEAN | default FALSE |
| `subscription_expires_at` | TIMESTAMPTZ | **nullable** — NULL = lifetime or never-paid |
| `subscription_status` | TEXT CHECK IN (active/cancelled/halted/expired) | nullable |
| `razorpay_subscription_id` | TEXT | filled when migrating to Subscriptions |
| `created_at` / `updated_at` | TIMESTAMPTZ | |

Partial index `WHERE subscription_expires_at IS NOT NULL AND tier != 'free'`
keeps the watchdog scan an index seek even at scale.

### 2.2 `usage_events` (metering v2, ADR 0001 step 1–2)

Append-only. Retention: 24 months (owner decision, ADR 0001 §7).

| column | type | notes |
|---|---|---|
| `id` | UUID PK | |
| `idempotency_key` | TEXT UNIQUE NOT NULL | `"{request_id}:{event_type}"` |
| `actor_kind` | TEXT CHECK IN (user/ext/ns/api_key/system) | |
| `actor_id` | TEXT NOT NULL | UUID str, ext uid, namespace, api_key_hash, or `"system"` |
| `user_id` | UUID FK → users ON DELETE SET NULL | null for anonymous/system |
| `api_key_id` | TEXT | SHA-256 truncated 16 hex chars — never the raw key |
| `namespace` | TEXT | integration namespace if applicable |
| `event_type` | TEXT NOT NULL | `write`, `read`, `encode`, `retrieve`, `knowledge`, `chat_message`, ... |
| `quantity` | INTEGER CHECK > 0 | default 1 |
| `unit_cost_cents` | INTEGER CHECK ≥ 0 | stamped at write time by `MeteringPricer` |
| `occurred_at` | TIMESTAMPTZ | event time |
| `recorded_at` | TIMESTAMPTZ | server insert time |
| `raw` | JSONB | event-specific extras (request_id, etc.) |

Indexes:
- `(actor_id, occurred_at DESC)` — per-actor queries (dashboards)
- `(event_type, occurred_at DESC)` — per-event-type rollups
- partial `(user_id, occurred_at DESC) WHERE user_id IS NOT NULL` — user stats
- `(occurred_at DESC)` — reconciler window scans

### 2.3 `metering_dead_letter`

Failed writes + reconciler drift findings. **The notifier IS the queue
consumer.** Rows where `notified_at IS NULL` are the pending backlog.

| column | type | notes |
|---|---|---|
| `id` | UUID PK | |
| `failed_at` | TIMESTAMPTZ DEFAULT NOW() | |
| `error_class` | TEXT | `ConnectionError`, `ReconciliationDrift`, etc. |
| `error_message` | TEXT | truncated to 500 chars |
| `payload` | JSONB | original event or finding |
| `retry_count` | INTEGER ≥ 0 | manual replay tooling uses this |
| `notified_at` | TIMESTAMPTZ | set by notifier once oncall is emailed |

Partial index `WHERE notified_at IS NULL` keeps the pump query cheap.

---

## 3 · Key classes — what they own, what they don't

### 3.1 `TierResolver` ([tier_resolver.py](../src/clsplusplus/tier_resolver.py))

```python
class TierResolver:
    def __init__(self, integration_store, cache_ttl_seconds: int = 300)
    def invalidate(self, api_key: Optional[str] = None) -> None
    async def resolve(self, api_key: str) -> Optional[str]
```

One instance per process, shared between `QuotaMiddleware` and
`MeteringPricer` so they **cannot disagree** about a user's tier.

- Cache hits: O(1), no DB.
- Cache misses: one join `integrations ⨝ api_credentials ⨝ users.email`.
- Exception from DB → returns `None` (caller decides: Quota falls back to `settings.tier`, Pricer returns 0).
- Cache hit TTL 5 min; invalidated from `/v1/user/upgrade` + from the watchdog per downgrade.

### 3.2 `MeteringWriter` ([metering_v2/writer.py](../src/clsplusplus/metering_v2/writer.py))

```python
@dataclass
class UsageEvent:
    idempotency_key: str
    actor_kind: str          # validated against VALID_ACTOR_KINDS
    actor_id: str
    event_type: str
    occurred_at: datetime
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    namespace: Optional[str] = None
    quantity: int = 1        # CHECK > 0 (raised in __post_init__)
    unit_cost_cents: int = 0 # CHECK ≥ 0
    raw: dict = field(default_factory=dict)

class MeteringWriter:
    async def record(self, event) -> None        # fire-and-forget
    async def record_sync(self, event) -> bool   # awaitable, for tests
```

`record()` uses `asyncio.create_task` so the caller's latency is zero.
On the background task:
- Happy path: `INSERT ... ON CONFLICT (idempotency_key) DO NOTHING`.
- Any failure: route the original payload to `metering_dead_letter`.
- If even the dead-letter insert fails: log loudly, event is LOST. That's
  an ops emergency monitoring must catch.

### 3.3 `MeteringPricer` ([metering_v2/pricing.py](../src/clsplusplus/metering_v2/pricing.py))

```python
def compute_unit_cost_cents(tier, event_type, current_ops, cap, rates) -> int
    # pure — no I/O. Used inside the class and directly from tests.

class MeteringPricer:
    async def price_event(self, api_key, event_type) -> int
```

Pricing rule (ADR 0001 §7 decision #2): flat tier covers ops ≤ cap; anything
beyond is charged per-event using `settings.overage_rates_cents`. Defaults
(pro=2¢, business=1¢, free/enterprise=0¢) ship in code and are overridable
via `CLS_OVERAGE_RATES_CENTS` (JSON env var).

### 3.4 `MeteringReconciler` ([metering_v2/reconciler.py](../src/clsplusplus/metering_v2/reconciler.py))

```python
async def reconcile_once(period=None) -> ReconciliationResult
def start() / stop()                       # daily loop (24h)
```

Compares `cls:ops:*:{period}` in Redis to `SUM(quantity)` grouped by
`api_key_id` in Postgres. Tolerances (configurable):

- `MIN_ABS_DRIFT = 5` (absolute ops) — ignore noise from async write window.
- `DRIFT_PERCENT = 0.001` — ADR 0001 §1.

Drift findings go into `metering_dead_letter` with
`error_class='ReconciliationDrift'`; the notifier emails oncall.

### 3.5 `MeteringNotifier` ([metering_v2/notifier.py](../src/clsplusplus/metering_v2/notifier.py))

```python
async def pump_once(self) -> int           # manual / test hook
def start() / stop()                       # 60s loop
```

- Polls 50 unnotified rows, groups by `error_class`, HTML-escapes payloads.
- Emails `settings.oncall_email` via `EmailService.send_metering_alert`.
- Email failure → `notified_at` **stays NULL**; next poll retries. No silent loss.
- If `CLS_ONCALL_EMAIL` unset → skip + log. Never marks rows notified for a no-op.

### 3.6 `SubscriptionWatchdog` ([subscription_watchdog.py](../src/clsplusplus/subscription_watchdog.py))

```python
async def run_once() -> {scanned, downgraded, errors}
def start() / stop()                       # 24h loop
```

- `get_expired_subscriptions(limit=500)` — partial-indexed scan.
- Per row: `expire_subscription(user_id)` sets `tier='free', status='expired'` with a `WHERE tier != 'free'` guard (safe reruns).
- Invalidates `TierResolver` per downgrade.
- Optional `notify(row)` callable — hook for a "your subscription expired" customer email (unused today; follow-up PR).

### 3.7 `MeteringHealthCheck` ([metering_v2/healthcheck.py](../src/clsplusplus/metering_v2/healthcheck.py))

Seven sub-checks, short-circuited by prerequisite:

```
config.flag_on                 (instant)
  ├─ config.oncall_email       (instant)
  └─ db.reachable              (SELECT 1)
        └─ db.schema_present   (information_schema scan)
              └─ writer.roundtrip    (canary INSERT + SELECT)
              └─ dead_letter.clean   (COUNT WHERE notified_at IS NULL AND >3m)
              └─ reconciler.drift    (reconcile_once: returns 0 drift)
```

Three entry points:
- `python -m clsplusplus.metering_v2 healthcheck` (CLI — exit code signal)
- `GET /admin/metering/health` (HTTP — 200/503 + JSON body)
- `pytest tests/test_metering_v2_healthcheck.py` (live, opt-in)

---

## 4 · Webhook payloads CLS++ understands

### 4.1 Razorpay (payment + subscription lifecycle)

| event | effect |
|---|---|
| `payment.captured` | `tier→paid`, `status='active'`, `expires_at=NOW()+30d` |
| `payment.failed` | log only |
| `subscription.charged` | `expires_at` = `current_end` (else `NOW()+30d`), `status='active'` |
| `subscription.cancelled` | `status='cancelled'`, **tier preserved** until watchdog expiry |
| `subscription.halted` | `tier='free'`, `status='halted'` — immediate |
| `subscription.completed` | `tier='free'`, `status='expired'` — immediate |
| any other event | log + ignore (no side effect) |

All handlers require valid `X-Razorpay-Signature` HMAC over the raw body.
Bad signature → raise `ValueError`. Missing `user_id` in `notes` → log + noop
(cannot route the update).

### 4.2 Stripe (parked)

`checkout.session.completed` → upgrade tier; `customer.subscription.deleted` → downgrade to free. Present for future re-activation; inactive in production today.

---

## 5 · Quota enforcement — detailed path

Before ADR fix, the quota middleware used the process-wide `settings.tier`.
After the fix, it reads each user's real tier from the DB.

```mermaid
flowchart TD
    A[Request arrives] --> B{enforce_quotas?}
    B -->|false| Z[pass through]
    B -->|true| C{public path or GET?}
    C -->|yes| Z
    C -->|no| D{has api_key?}
    D -->|no| Z
    D -->|yes| E[TierResolver.resolve api_key]
    E --> F{tier str}
    F -->|None| G[fallback: settings.tier]
    F -->|str| H[Tier enum]
    G --> H
    H --> I[check_quota key, tier, settings]
    I --> J{allowed?}
    J -->|yes| Z
    J -->|no| K[HTTP 402 tier, usage, limit]
    I -.->|Redis error| L{quota_fail_closed?}
    L -->|yes default| M[HTTP 503 + Retry-After: 5]
    L -->|false opt-in| Z
```

---

## 6 · Observability surfaces

| Signal | Where | Audience |
|---|---|---|
| `cls:metrics:{user}:{period}` | Redis HASH | admin dashboard DAU/MAU |
| `cls:active:global` | Redis ZSET (15-min window) | landing "active now" widget |
| `cls:dau:global:{YYYY-MM-DD}` | Redis SET | admin + waitlist promoter |
| `usage_events` | Postgres | billing source of truth (ADR 0001) |
| `metering_dead_letter` | Postgres | notifier → oncall email |
| `X-Request-Id` header | every response | cross-service trace correlation |
| `/v1/memory/traces` | in-memory tracer | latency / call-graph debugging |
| `/admin/metering/health` | HTTP | uptime monitor + admin UI + CI |
| `/admin/metering/reconcile` | HTTP | on-demand drift check |
| `/admin/subscriptions/expire-due` | HTTP | manual watchdog trigger |

---

## 7 · Environment variables (billing-critical only)

See `.env.example` for the full list. Essentials:

| var | default | purpose |
|---|---|---|
| `CLS_DATABASE_URL` | postgres://cls:cls@localhost:5432/cls | primary SoT |
| `CLS_REDIS_URL` | redis://localhost:6379 | hot counters |
| `CLS_JWT_SECRET` | `""` | user session signing (required in prod) |
| `CLS_ENFORCE_QUOTAS` | `true` | hard-block at cap |
| `CLS_QUOTA_FAIL_CLOSED` | `true` | return 503 on Redis error |
| `CLS_METERING_V2_WRITE_ENABLED` | `false` | dual-write flag |
| `CLS_ONCALL_EMAIL` | `rjabbala@gmail.com` | dead-letter digest |
| `CLS_OVERAGE_RATES_CENTS` | `{pro:2¢, business:1¢}` | pay-as-you-go rates |
| `CLS_GOOGLE_CLIENT_ID/SECRET/REDIRECT_URI` | `""` | OAuth |
| `CLS_GITHUB_CLIENT_ID/SECRET/REDIRECT_URI` | `""` | OAuth |
| `CLS_FRONTEND_URL` | `""` (`https://www.clsplusplus.com` in prod) | post-OAuth redirect host |
| `CLS_COOKIE_DOMAIN` | `""` (`.clsplusplus.com` in prod) | session cookie scope |
| `CLS_RAZORPAY_KEY_ID/SECRET/WEBHOOK_SECRET` | `""` | billing |
| `CLS_RESEND_API_KEY` | `""` | transactional email |

---

## 8 · Test classification

Every test in `tests/` belongs to exactly one stage. The same classification
drives the admin Testing page.

| Stage | Scope | Infra required | Runs where |
|---|---|---|---|
| **Unit** | pure functions + classes with fake I/O | none | GH Actions CI (every push/PR) + locally |
| **Integration — DB** | `tests/test_*_ddl` + `schema`, round-trip inserts | Postgres | GH Actions CI (with Postgres service) |
| **Integration — HTTP** | `tests/test_tiers.py::TestQuotaMiddleware` etc. — full FastAPI with ASGI client | Redis + Postgres | GH Actions CI |
| **Smoke — live** | `tests/test_billing_e2e.py` — hits the real deployed API | live URL + admin creds | `prod-smoke.yml` every 15 min |
| **Healthcheck — live** | `/admin/metering/health` via HTTP | live URL + admin cookie | on-demand, hourly uptime monitor |
| **Playwright — UI** | `v0-cls-memory-website/tests/e2e/*.spec.ts` | local dev server or live URL | locally + scheduled workflow |

See [the admin Testing page](../v0-cls-memory-website/app/admin/tests/page.tsx) for the interactive view.

---

## 9 · Deploy pipeline (how a change reaches prod)

```mermaid
sequenceDiagram
    autonumber
    participant Dev
    participant GH as GitHub
    participant CI as ci.yml
    participant Render
    participant Vercel
    participant Smoke as prod-smoke.yml

    Dev->>GH: push branch + open PR
    GH->>CI: run unit tests (matrix 3.10/3.11/3.12, PG + Redis services)
    CI-->>GH: green
    Dev->>GH: merge to main
    par Render auto-deploy
        GH->>Render: webhook → docker build → new container
        Render-->>Render: health check
        Render-->>GH: deploy succeeded
    and Vercel auto-deploy (frontend monorepo)
        GH->>Vercel: webhook → pnpm build → CDN
    end
    GH->>Smoke: on push to main → run E2E suite vs prod
    Smoke->>Smoke: login as CI admin, run 16 tests
    Smoke-->>GH: green (or red with /metering/health + /metering/reconcile dump)
    alt failure
        Smoke-->>Dev: email + optional Slack
    end
```

---

## 10 · Known gaps (tracked, not blocking)

- **Razorpay is one-time Orders today**, not recurring Subscriptions. Users re-pay every 30 days (or the watchdog downgrades them). Webhook handlers for `subscription.*` are live but idle until migration. Recorded in ADR follow-ups.
- **No customer-facing "subscription expired" email** yet. `SubscriptionWatchdog.notify` hook exists; receiver needs wiring.
- **Pricing rates** ship with conservative defaults; finalise with real usage data.
- **Metering v2 read path** (ADR 0001 step 4+) — Redis is still primary for `check_quota` and `/v1/user/usage` until we cut over.

# CLS++ — High Level Design

**Status:** living document · last major update 2026-04-22

This is the system-level view. For component internals, see [LLD.md](LLD.md).
For individual architectural decisions, see `docs/adr/`.

---

## 1 · What CLS++ is

CLS++ is a **cross-LLM memory-as-a-service**. Customers install a Chrome
extension or use the SDK / CLI; their chat context with one model (Claude,
GPT, Gemini) is captured, consolidated through a phase-engine, and
transparently re-injected when they switch to a different model. The
service is billed per monthly operation count, with a flat tier + pay-as-you-go
overage pricing model.

---

## 2 · Deployment shape

```
               ┌──────────────────────────┐
Users ──────►  │  Vercel (Next.js 16)     │  https://www.clsplusplus.com
  (browser,    │  • marketing site        │
   Chrome      │  • signup / login        │
   extension,  │  • /profile, /admin      │
   SDK, CLI)   │  • /api/* → rewrite      │
               └────────────┬─────────────┘
                            │ server-side rewrite
                            ▼
               ┌──────────────────────────┐
               │  Render (FastAPI)        │  https://clsplusplus.onrender.com
               │  • memory engine         │
               │  • billing + metering    │
               │  • admin APIs            │
               │  • OAuth callbacks       │
               └────────────┬─────────────┘
                ┌───────────┼────────────┬─────────────┐
                ▼           ▼            ▼             ▼
           ┌────────┐  ┌───────┐  ┌──────────┐  ┌─────────┐
           │Postgres│  │ Redis │  │  MinIO   │  │ Resend  │
           │(pgvec) │  │(rate  │  │ (L3 +    │  │ (email) │
           │        │  │ limit,│  │ cls-     │  │         │
           │        │  │ ops   │  │ metering)│  │         │
           │        │  │ cntr) │  │          │  │         │
           └────────┘  └───────┘  └──────────┘  └─────────┘
```

**Key architectural decisions** (recorded in ADRs):

- `api.clsplusplus.com` does **not** serve the UI. Vercel owns `www.` and
  server-side rewrites `/api/v1/*` and `/api/admin/*` to the Render host.
  This is why OAuth redirect URIs must register on `www.` not `api.`
  ([ADR 0001 §Context](adr/0001-metering-data-lake.md)).
- Session cookies use `Domain=.clsplusplus.com` so the cookie set by the
  backend (delivered through the Vercel proxy) is valid on both `www.` and
  any future subdomains. See `CLS_COOKIE_DOMAIN` in config.
- Metering is moving from a Redis-only counter pipeline to an append-only
  durable event log in Postgres, with Parquet archives on MinIO for ≥30-day
  history. Implementation is staged 1→8; steps 1–3 plus healthcheck are
  live (see [ADR 0001 §6 rollout](adr/0001-metering-data-lake.md)).

---

## 3 · Top-level flows

The next sections give a high-level sequence for each major flow. Low-level
details, class diagrams, and error paths are in [LLD.md](LLD.md).

### 3.1 · User onboarding (email or OAuth)

```mermaid
sequenceDiagram
    autonumber
    participant Browser
    participant Vercel as Vercel (UI + proxy)
    participant API as Render API
    participant OAuth as Google / GitHub
    participant DB as Postgres (users)
    participant Mail as Resend

    alt Email + password signup
        Browser->>Vercel: POST /api/v1/auth/register
        Vercel->>API: POST /v1/auth/register (proxied)
        API->>DB: INSERT pending_registrations (OTP hash, expiry)
        API->>Mail: send_verification_email(OTP, magic link)
        API-->>Vercel: {pending: true}
        Browser->>Vercel: submit OTP at /verify-email
        Vercel->>API: POST /v1/auth/verify-register
        API->>DB: INSERT users (email, password_hash, email_verified=true)
        API-->>Browser: Set-Cookie cls_session; redirect /profile
    else OAuth (Google or GitHub)
        Browser->>Vercel: click "Continue with Google"
        Browser->>Vercel: GET /api/v1/auth/google?redirect=/profile
        Vercel->>API: GET /v1/auth/google
        API-->>Browser: 307 → accounts.google.com/...&redirect_uri=www/.../callback
        Browser->>OAuth: consent
        OAuth-->>Browser: 302 → /api/v1/auth/google/callback?code=...
        Browser->>Vercel: GET /api/v1/auth/google/callback?code=...
        Vercel->>API: proxied GET
        API->>OAuth: exchange code (client_secret)
        OAuth-->>API: {access_token, id_token}
        API->>OAuth: GET /userinfo
        API->>DB: UPSERT users (link google_id, auto-verify email)
        API-->>Browser: Set-Cookie cls_session; 302 → www.clsplusplus.com/profile
    end
```

### 3.2 · Authenticated API call → metering → quota enforcement

Every memory API call goes through this path. The durable event log is the
safety-net source of truth; the Redis counter is the hot path today.

```mermaid
sequenceDiagram
    autonumber
    participant Client as Extension / SDK / CLI
    participant API as Render API
    participant Mw as AuthMiddleware
    participant Quota as QuotaMiddleware
    participant Redis
    participant PG as Postgres (usage_events)
    participant DL as metering_dead_letter

    Client->>API: POST /v1/memory/write  (Bearer cls_live_...)
    API->>Mw: resolve api_key
    Mw->>PG: JOIN integrations/api_credentials → namespace + owner
    Mw-->>API: request.state.api_key + namespace

    API->>Quota: should this request run?
    Quota->>Redis: GET cls:ops:{key}:{YYYY-MM}
    alt under cap
        Quota-->>API: allow
    else over cap
        Quota->>PG: resolve_tier_from_key (TierResolver, 5-min cache)
        Quota-->>Client: 402 {tier, usage, limit}
    end

    API->>API: handle /v1/memory/write
    API->>Client: 200 OK

    par fire-and-forget metering
        API->>Redis: HINCRBY cls:usage:{key} + INCR cls:ops:{key}
        API->>PG: INSERT usage_events (idempotency_key, unit_cost_cents, ...)
        Note right of PG: ON CONFLICT DO NOTHING so retries collapse
    and fallback on write failure
        API->>DL: INSERT metering_dead_letter (payload, error_class)
    end
```

### 3.3 · Payment, tier upgrade, subscription expiry

All paid tiers now stamp `subscription_expires_at`. The watchdog runs daily
and auto-downgrades rows whose window has elapsed.

```mermaid
sequenceDiagram
    autonumber
    participant Browser
    participant API as Render API
    participant RP as Razorpay
    participant DB as Postgres
    participant WD as SubscriptionWatchdog (24h)
    participant TR as TierResolver cache

    rect rgba(60, 90, 120, 0.18)
    note over Browser,DB: Payment path (today: one-time Orders)
    Browser->>API: POST /v1/billing/create-order (tier=pro)
    API->>RP: client.order.create(amount, notes={user_id, tier})
    RP-->>API: order_id
    API-->>Browser: order + key_id
    Browser->>RP: complete checkout
    RP->>API: webhook payment.captured (signed)
    API->>DB: UPDATE users SET tier=pro,\n  subscription_expires_at=NOW()+30d,\n  subscription_status='active'
    API->>TR: invalidate cache (next request sees new tier)
    end

    rect rgba(90, 120, 60, 0.18)
    note over WD,DB: Expiry path (daily)
    WD->>DB: SELECT id,email,tier FROM users\n  WHERE subscription_expires_at < NOW()\n    AND tier != 'free'
    loop each expired user
        WD->>DB: UPDATE users SET tier='free', subscription_status='expired'
        WD->>TR: invalidate cache
    end
    end

    rect rgba(120, 60, 60, 0.18)
    note over RP,DB: Future: recurring Subscriptions (handlers wired today)
    RP->>API: webhook subscription.charged (current_end=...)
    API->>DB: extend expires_at to current_end
    RP->>API: webhook subscription.cancelled
    API->>DB: status='cancelled', tier preserved until expiry
    RP->>API: webhook subscription.halted
    API->>DB: tier='free' immediately, status='halted'
    end
```

### 3.4 · Metering safety loop (reconciler + notifier)

The reconciler proves the durable log and the Redis counter agree. Any drift
turns into a `metering_dead_letter` row, the notifier pages on-call.

```mermaid
sequenceDiagram
    autonumber
    participant RC as MeteringReconciler (24h)
    participant Redis
    participant PG as Postgres
    participant DL as metering_dead_letter
    participant NF as MeteringNotifier (60s)
    participant Mail as Resend
    participant Oncall as on-call agent

    loop every 24h
        RC->>Redis: SCAN cls:ops:*:{period}
        RC->>PG: SELECT api_key_id, SUM(quantity)\n FROM usage_events WHERE period
        RC->>RC: compare per api_key_hash
        alt drift > 0.1% and > 5 abs
            RC->>DL: INSERT (error_class='ReconciliationDrift', payload JSON)
        end
    end

    loop every 60s
        NF->>DL: SELECT WHERE notified_at IS NULL LIMIT 50
        alt batch non-empty
            NF->>Mail: send_metering_alert(oncall, digest html)
            Mail-->>Oncall: email
            NF->>DL: UPDATE notified_at = NOW()
        end
    end
```

### 3.5 · Health self-check (on-demand or monitoring)

A single endpoint answers "is metering alive?" with seven sub-checks. Used
by CI, by humans, and wired to the admin dashboard.

```mermaid
sequenceDiagram
    autonumber
    participant Caller as CI / curl / admin UI
    participant API as /admin/metering/health
    participant HC as MeteringHealthCheck
    participant Redis
    participant PG as Postgres
    participant DL as metering_dead_letter
    participant RC as Reconciler
    participant WR as MeteringWriter

    Caller->>API: GET
    API->>HC: run_all()
    HC->>HC: config.flag_on + config.oncall_email
    HC->>PG: SELECT 1 (db.reachable)
    HC->>PG: check tables exist (db.schema_present)
    HC->>WR: record_sync(canary event_type='healthcheck')
    HC->>PG: SELECT where idempotency_key=<canary> (writer.roundtrip)
    HC->>DL: COUNT(*) WHERE notified_at IS NULL > 3m (dead_letter.clean)
    HC->>RC: reconcile_once() (reconciler.drift)
    HC-->>API: HealthReport{passed, checks[]}
    alt passed
        API-->>Caller: 200 + JSON
    else any check failed
        API-->>Caller: 503 + JSON (each check's remediation)
    end
```

---

## 4 · Cross-cutting concerns

| Concern | Primary mechanism | Fallback / safety |
|---|---|---|
| Auth | `AuthMiddleware`: Bearer API key or `cls_session` JWT cookie | 401 if both fail and `CLS_REQUIRE_API_KEY=true` |
| Rate limit | `RateLimitMiddleware`: per-key sliding-window in Redis | fails open if Redis down (documented trade-off) |
| Quota | `QuotaMiddleware`: per-user tier via `TierResolver` | fail-**closed** (503 + Retry-After) by default; override with `CLS_QUOTA_FAIL_CLOSED=false` |
| Metering write | `MeteringWriter.record()` fire-and-forget → `usage_events` | fallback to `metering_dead_letter` + oncall email |
| Pricing | `MeteringPricer.price_event()` — stamps `unit_cost_cents` at write time | defaults to 0 when `CLS_OVERAGE_RATES_CENTS` unset = no accidental bill |
| Subscription | `payment.captured` → set `expires_at = NOW()+30d` | `SubscriptionWatchdog` scans daily for elapsed windows → downgrade |
| OAuth | Redirects go through Vercel proxy so Set-Cookie lands on `.clsplusplus.com` | `CLS_*_REDIRECT_URI` env vars override the computed URI |
| Observability | Structured logs + dead-letter paging | `/admin/metering/health` exposes everything in one JSON |

---

## 5 · Sources of truth (SoT)

| Domain | SoT today | SoT after ADR 0001 fully lands |
|---|---|---|
| User identity | `users` table | unchanged |
| API keys | `api_credentials` (hashed) | unchanged |
| Tier (paid) | `users.tier` | unchanged |
| Subscription window | `users.subscription_expires_at` | unchanged |
| Per-API-key ops count (this month) | Redis `cls:ops:{key}:{period}` | `usage_events` (step 5) |
| Per-API-key usage history | Redis `cls:usage:{key}:{period}` + monthly_metrics | `usage_events` + Parquet on MinIO (step 6) |
| Billing reconciliation | none | daily reconciler diff (step 3 — live) |

---

## 6 · Security model (summary)

- **Secrets** live only in Render env / Vercel env / GitHub Actions secrets — never in code.
- **Admin endpoints** are gated on `request.state.is_admin` which is set only when a valid JWT carrying `is_admin=true` is presented.
- **API keys** are SHA-256 hashed at rest; only prefixes are logged.
- **PII in telemetry**: extension `site` URLs are treated as PII — hashed at write time (owner decision in ADR 0001).
- **Webhook signatures** are HMAC-verified before any side effect; bad signature → raise, never silently process.
- **Fail-closed** on quota check errors by default — billing correctness > availability during Redis outages.

---

## 7 · Regression gates

Three independent layers:

1. **Unit tests** — `ci.yml` runs `pytest tests/` on every push + PR across Python 3.10 / 3.11 / 3.12 with Postgres + Redis services. Full suite: 161+ tests.
2. **Live E2E** — `prod-smoke.yml` runs `tests/test_billing_e2e.py` against production every 15 min, on every push to `main`, and on manual dispatch. 16 tests covering the whole billing spine.
3. **Healthcheck endpoint** — `/admin/metering/health` is cheap enough to wire to an uptime monitor (503 on failure).

If all three are green, the metering + billing pipeline is behaving correctly end-to-end.

---

## 8 · What's next

Tracked in ADR 0001, step by step:

- [ ] Step 4 — `MeteringQuery` read interface (UI + `check_quota` read from `usage_events` not Redis)
- [ ] Step 5 — cut reads over behind per-endpoint flag
- [ ] Step 6 — Parquet rollup daily → MinIO (the "data lake" proper)
- [ ] Step 7 — backfill 6 months of Redis history
- [ ] Step 8 — retire Redis TTLs

Other forward-looking items:

- Migrate Razorpay client from **one-time Orders** to **recurring Subscriptions** (webhook handlers already wired).
- Grace-period policy on failed payments (currently: immediate halt on `subscription.halted`).
- Customer-facing "subscription expired" email (hook exists on `SubscriptionWatchdog.notify`; receiver TBD).

# ADR 0001: Metering pipeline with a data-lake backbone

## Status

**Proposed** — awaiting owner review. No code changes accompany this ADR;
every production path still runs the current Redis-only pipeline.

## Good-morning summary

Current metering is **a Redis-only, fire-and-forget pipeline with 35–65 day
TTLs, no audit trail, no reconciliation against Razorpay/Stripe invoices, and
silently-swallowed exceptions**. If Redis is unhealthy for a week or a counter
expires before invoicing, we bill the wrong amount and have no way to prove
what actually happened. This ADR proposes an **append-only event log as the
single source of truth**, with Redis demoted from "storage" to "read-cache",
and a standard read interface the UI and billing both consume. Recommended
option is the **mid path** (§ Option B). Open questions in § 7.

---

## 1 · Context

### What exists today (file:line refs, not guesses)

**Write path.** Every metering call is fire-and-forget into Redis:

- [usage.py:26](src/clsplusplus/usage.py:26) `record_usage(api_key, op)` → `HINCRBY cls:usage:{key}:{YYYY-MM}` (35-day TTL).
- [usage.py:44](src/clsplusplus/usage.py:44) `record_operation(api_key)` → `INCR cls:ops:{key}:{YYYY-MM}` (35-day TTL). **This is the counter billing reads.**
- [metrics.py:42](src/clsplusplus/metrics.py:42) `MetricsEmitter.emit(user, metric)` → `HINCRBY cls:metrics:{user}:{YYYY-MM}` (65-day TTL).
- [metrics.py:106](src/clsplusplus/metrics.py:106) `record_active_user(id)` → `ZADD cls:active:global` (1 h) + `SADD cls:dau:global:{YYYY-MM-DD}` (35 d).
- [metrics.py:149](src/clsplusplus/metrics.py:149) `record_ext_telemetry` → nine separate keys with 35–95 day TTLs.
- [api.py:231–2665](src/clsplusplus/api.py:231) — 15 call sites of `_record_usage` on every mutating/reading memory op.
- Every function ends in `except Exception: pass`. A Redis outage drops usage silently.

**Read path.** UI and billing both read Redis directly:

- [api.py:1427](src/clsplusplus/api.py:1427) `/v1/user/usage` → `get_quota_status` → Redis.
- [api.py:1452](src/clsplusplus/api.py:1452) `/v1/user/usage/history` → `get_usage_history` → six-month Redis scan.
- [tiers.py:67](src/clsplusplus/tiers.py:67) `check_quota` → Redis (the **enforcement hot path**; returns 402 when exceeded).

**Billing tie-in.** [tiers.py:46](src/clsplusplus/tiers.py:46) fixed monthly prices (₹0 / $9 / $29 / $149) multiplied by a tier selection in [api.py:1441](src/clsplusplus/api.py:1441). **Nothing reconciles payment amounts against observed operation counts.** If a Business user consumes 5 M ops (25× the 200 k quota ceiling) we still invoice $29 because quota enforcement only blocks once the counter clears `ops_per_month`.

### Forcing function

Owner statement: "if you fuck up we lose money." The concrete failure modes already latent in the code:

1. **Data loss.** Redis unhealthy > 35 days before invoicing → counters evict → user is billed from a zero baseline.
2. **No audit.** A disputed invoice cannot be proved against the ledger because there is no ledger.
3. **Silent drops.** `except: pass` means we cannot distinguish "user did nothing" from "Redis was down".
4. **Race conditions.** `INCR` + `check_quota` is not atomic; a burst can exceed the quota by the entire request concurrency.
5. **No replay.** Bug in counter logic → no way to recompute historical billing from source events.

### Hard constraints

- Billing-grade retention: event record must survive at least 18 months (tax + dispute window).
- Idempotent writes: a request retry must not double-count.
- Reconcilable: every invoice line item traces to a measurable set of events.
- Reversible rollout: old and new must coexist until cut-over is proven.

---

## 2 · Decision (proposed)

Add an **append-only event log** as the single source of truth for metering. Every `_record_usage`, `emit`, `record_active_user`, `record_ext_telemetry` call writes a JSON line to the log. Redis stays in place as a **hot aggregation cache** rebuilt from the log, not a primary store. UI and billing reconcile via a single read interface (`MeteringQuery`) that reads the aggregation cache but can transparently fall back to re-aggregating from the log for any period.

---

## 3 · Alternatives considered

### Option A — Minimal: Postgres append-only table

Add `usage_events` table (event_id PK, idempotency_key UNIQUE, user_id, api_key, event_type, quantity, created_at, raw JSONB). Dual-write from each call site; the existing Redis calls remain. Quota still reads Redis for the hot path; billing reads from Postgres.

**Good at:** zero new infra; ships in days; Postgres already holds users/integrations; pgBackRest already runs.
**Bad at:** Postgres at 100+ events/sec/tenant needs partitioning; analytic queries compete with OLTP; no columnar compression.
**Cost:** ~1 person-week to land dual-write + backfill. Storage: ~200 MB/month/10 k users.
**Revisit if:** event rate crosses ~1 M/day.

### Option B — Mid: Parquet on S3 (MinIO locally, Render-hosted S3 in prod) + Postgres hot table (RECOMMENDED)

Writes go to Postgres `usage_events` first (durable, idempotent via `idempotency_key`). A rollup worker compacts daily → monthly Parquet objects on S3, partitioned by `period/tenant`. UI queries via `MeteringQuery` abstraction: recent (≤ 30 days) reads from Postgres; historical reads from Parquet via DuckDB (embeddable, no separate service).

**Good at:** durable-first write → cheap columnar storage for old data → same read interface covers both. No new runtime service to operate; DuckDB is a library.
**Bad at:** two-tier read path is more code than option A; MinIO already in stack reduces the deployment burden but adds one moving part.
**Cost:** ~2–3 person-weeks to land. Storage: Parquet compresses ~10× vs JSON; billing event archive is < 1 GB/year at current scale.
**Revisit if:** query latency on 12-month windows exceeds 2 s, or event rate > 10 M/day (then move to ClickHouse).

### Option C — Heavy: ClickHouse or Timescale

Dedicated time-series OLAP server. Handles the schema migration problem and real-time analytic queries natively.

**Good at:** single system for all analytical reads; handles 10× scale.
**Bad at:** an additional always-on service to operate, backup, and secure. Overhead disproportionate to our current volume (thousands of events/day per tenant, not millions).
**Cost:** ~4–6 person-weeks. Minimum $30–50/month hosted.
**Revisit if:** we genuinely hit the Option B ceiling.

### Do-nothing

Keep Redis-only with 35-day TTLs. **Not acceptable** — all five failure modes in § 1 are unaddressed.

---

## 4 · Tradeoffs

| Axis | A (PG) | **B (PG + S3/Parquet)** | C (ClickHouse) |
|---|---|---|---|
| Implementation | 1 wk | **2–3 wk** | 4–6 wk |
| Ops burden (pages/qtr) | Low | **Low** | Medium |
| Billing-audit grade | Good | **Excellent** (18-mo retention cheap) | Excellent |
| Scale ceiling | ~1 M events/day | **~50 M events/day** | ~1 B/day |
| Reversibility if wrong | Easy (drop table) | **Easy (Postgres writer is the primary; S3 is cache)** | Hard |
| Cost @ current scale | $0 | **~$2/mo MinIO, else $0** | $30–50/mo |

**Decision driver: reversibility × retention.** Option B makes the durable writer the same Postgres we already run, so the rollout has no point of no return beyond the schema migration. The Parquet/DuckDB read path is additive; if it fails we stay on the Postgres hot path until it's fixed.

---

## 5 · Consequences

**What becomes easier**
- Billing reconciliation: every invoice cites a deterministic event window; disputes take minutes instead of days.
- Replayable billing: a bug in counter logic can be fixed retroactively by re-rolling from the event log.
- Observability: silent failure goes away; write failures raise to a dead-letter table.

**What becomes harder**
- Two-tier read (hot Postgres vs cold Parquet) is more code than today's single Redis fetch. Mitigated by the `MeteringQuery` abstraction.
- Schema migration for `usage_events` needs care (it's the table billing reads).

**What we are explicitly accepting**
- Postgres write overhead (~1 ms extra per metered call). Fire-and-forget will wrap it so latency at the edge does not regress.
- One additional background worker (the rollup), already a cron-pattern in the repo.

**Revisit if**
- Event rate exceeds 10 M/day.
- Parquet query latency on 12-month windows > 2 s.
- We add a second product that needs its own metering schema.

**What this ADR is NOT**
- Not a billing-logic rewrite. Prices, tiers, and webhook handlers stay untouched.
- Not a switch away from Redis for real-time ("active now", DAU). That role stays.
- Not a proposal to retire the current `MetricsEmitter` API — it becomes a thin wrapper over the new writer.

---

## 6 · Rollout (no point-of-no-return until step 6)

1. **Land the schema** (`usage_events`, `metering_dead_letter`) behind a feature flag; no writers yet. *Reversible: drop the tables.*
2. **Dual-write** from `record_usage`, `record_operation`, `MetricsEmitter.emit`, `record_active_user`, `record_ext_telemetry`. Redis still primary. Feature flag gates the new writer only. *Reversible: flip the flag off.*
3. **Reconcile for a week**: daily job that compares Postgres aggregates to Redis counters, alerts on > 0.1% drift. Expected: zero drift.
4. **Land `MeteringQuery` read interface.** `get_quota_status`, `check_quota`, `get_usage_history` rewritten to use it. Backed by Redis for now — read surface stabilises before cut-over.
5. **Switch reads** to Postgres. Redis stays as a cache. Monitor drift daily.
6. **Land Parquet rollup** (daily → monthly → S3). DuckDB read for historical windows. **This is the point-of-no-return: deleting Parquet loses cheap archives.**
7. **Backfill** 6 months of history by re-aggregating from Redis where possible (best-effort; document gaps).
8. **Retire Redis TTLs** (keep Redis itself). Event log is now the system of record.

Each step ships independently and is individually reversible. The cut-over from Redis-reads to Postgres-reads (step 5) is the most delicate and will be behind a per-endpoint flag.

---

## 7 · Open questions (owner input required)

1. **Retention.** 18 months or longer for tax/dispute? Defaults to 18 unless you say otherwise.
2. **Prices tied to usage?** Currently flat monthly tiers. Should we charge on over-quota usage (pay-as-you-go above tier cap)? This changes the event schema (need a "unit cost" attached per event).
3. **Data-lake host.** MinIO is already in the stack ([minio_endpoint](src/clsplusplus/config.py:53)). Is that acceptable for production invoice-grade data, or do we need an external S3-compatible provider?
4. **Privacy in the log.** Some events today record `site` for extension telemetry. Is the site URL PII for your jurisdiction? If yes, we hash it in the event.
5. **Dead-letter policy.** When Postgres write fails, we queue to `metering_dead_letter` and alert. Who gets paged?
6. **Back-dated invoices.** How far back do we need to be able to re-issue an invoice if we find a counter bug?

No implementation code will be written until these six questions are resolved and the ADR status moves from **Proposed** to **Accepted**.

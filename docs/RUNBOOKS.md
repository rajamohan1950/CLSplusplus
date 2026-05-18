# Operations Runbooks

Emergency procedures for the CLS++ API (FastAPI on Render). Keep this short and
actionable — when something is on fire, scan, act, verify.

Related docs: [LAUNCH_CHECKLIST.md](../LAUNCH_CHECKLIST.md),
[DB_MIGRATIONS.md](DB_MIGRATIONS.md), [DEPLOY_RENDER.md](DEPLOY_RENDER.md).

---

## 1. Health check failing on Render

**Symptom:** Render marks the service unhealthy / restarts it; `/health` returns
non-200 or times out.

1. Open the Render dashboard → service → **Logs**. Look for a crash on startup
   (missing env var, DB/Redis unreachable).
2. Confirm `/health` is reachable directly:
   `curl -i https://www.clsplusplus.com/health`. It is a public path
   ([middleware.py](../src/clsplusplus/middleware.py)) and must return `200`.
3. If `/health` is 403, the abuse guard is blocking the Render proxy IP — check
   `CLS_ABUSE_GUARD_ENABLED` and the blocklist; `/health` and `/v1/health` are
   exempt by design, so a 403 here is a regression.
4. If startup is crashing: verify required secrets are set (`CLS_JWT_SECRET`,
   `CLS_DATABASE_URL`, `CLS_REDIS_URL`) — see LAUNCH_CHECKLIST.md §1.
5. If DB/Redis is the cause, check those Render services are running and the
   internal connection URLs are current.
6. Roll back: Render dashboard → **Deploys** → redeploy the last known-good
   commit.

## 2. Database schema rollback

**Symptom:** A schema change broke production.

1. Stop further writes if the change is actively corrupting data (scale the
   service to 0 instances on Render, or take it out of rotation).
2. Apply the reverse SQL. Every migration in `scripts/migrations/NNN_*.sql`
   carries its rollback block as a trailing comment — run it:
   `psql "$CLS_DATABASE_URL" -f <(rollback SQL)`.
3. If there is no migration file (ad-hoc change), reconstruct the reverse
   `ALTER` by hand. See [DB_MIGRATIONS.md](DB_MIGRATIONS.md).
4. Redeploy the code commit that matches the rolled-back schema.
5. Verify with a smoke request against the affected endpoint.

## 3. Circuit breaker tripped (Razorpay / GeoIP / Resend / demo LLM)

**Symptom:** A flaky external dependency causes fast failures; logs show a
breaker in the OPEN state (see `src/clsplusplus/resilience.py`).

1. Identify which breaker: logs name the call site (Razorpay billing, GeoIP
   lookup, Resend email, demo LLM).
2. A tripped breaker is **working as designed** — it fails fast for the cooldown
   (`CLS_CIRCUIT_RECOVERY_SECONDS`, default 30s) then probes once.
3. Check the upstream provider's status page. If the provider is down, wait —
   the breaker will close itself once a probe succeeds.
4. User impact by breaker:
   - **GeoIP** — fails open; new signups are treated as allowed. No action.
   - **Resend email** — verification/invite emails are not sent; signup stalls.
     Tell affected users to retry once the provider recovers.
   - **Razorpay** — checkout/billing is degraded; free-tier usage is unaffected.
   - **demo LLM** — only the public demo endpoints degrade.
5. If a breaker is stuck open after the provider has recovered, restart the
   service to reset breaker state.

## 4. Metering dead-letter replay / purge

**Symptom:** The metering reconciler found drift and enqueued rows into
`metering_dead_letter`; the on-call digest email fired.

1. Check pipeline health: `GET /admin/metering/health` (admin session required).
   It reports dead-letter count and reconciler drift.
2. Inspect the queued rows:
   `psql "$CLS_DATABASE_URL" -c "SELECT * FROM metering_dead_letter ORDER BY created_at DESC LIMIT 50;"`
3. Re-run the reconciler for the affected period:
   `POST /admin/metering/reconcile?period=YYYY-MM` (admin session). This is
   idempotent and re-enqueues only genuine drift.
4. If a dead-letter row is a confirmed false positive (already corrected),
   delete it by id after recording why in the deploy log.
5. Metering v2 is feature-flagged (`CLS_METERING_V2_WRITE_ENABLED`); if it is
   off, the dead-letter table is inert and rows can be ignored.

---

## Escalation

On-call address for metering pager digests: see `CLS_ONCALL_EMAIL`
(`src/clsplusplus/config.py`). For anything customer-billing related, prefer
fixing forward over guessing — the metering pipeline is idempotent and safe to
re-run.

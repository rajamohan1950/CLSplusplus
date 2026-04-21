-- CLS++ Metering v2 — append-only event log and dead-letter queue.
-- Background: docs/adr/0001-metering-data-lake.md
--
-- This DDL is ONLY applied when CLS_METERING_V2_WRITE_ENABLED is true.
-- No production code writes to these tables yet.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- -----------------------------------------------------------------------------
-- usage_events : the source of truth
-- -----------------------------------------------------------------------------
-- Every metered action writes exactly one row here. Rows are never updated
-- or deleted in normal operation; retention is 24 months per ADR decision.
--
-- `actor_kind` + `actor_id` unifies every identity we track today:
--   user  → users.id (UUID as text)
--   ext   → extension uid
--   ns    → namespace string
--   api_key → an api-key hash (never the raw key)
--   system → aggregate / job-level events
--
-- `unit_cost_cents` exists because pricing is hybrid: flat tier until the
-- cap, then pay-as-you-go. The writer stamps the cost at the moment the
-- event happened — not at invoice time — so a later tier change or price
-- update never alters historical invoices.
--
-- `raw` carries event-specific extras (e.g. model_name, tokens, site_hash).
-- Raw site URLs MUST be hashed before being put in here (owner decision:
-- treating site URLs as PII).

CREATE TABLE IF NOT EXISTS usage_events (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key   TEXT         NOT NULL UNIQUE,
    actor_kind        TEXT         NOT NULL
                                   CHECK (actor_kind IN ('user', 'ext', 'ns', 'api_key', 'system')),
    actor_id          TEXT         NOT NULL,
    user_id           UUID         REFERENCES users(id) ON DELETE SET NULL,
    api_key_id        TEXT,
    namespace         TEXT,
    event_type        TEXT         NOT NULL,
    quantity          INTEGER      NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit_cost_cents   INTEGER      NOT NULL DEFAULT 0 CHECK (unit_cost_cents >= 0),
    occurred_at       TIMESTAMPTZ  NOT NULL,
    recorded_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    raw               JSONB        NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_usage_events_actor_time
    ON usage_events(actor_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_events_type_time
    ON usage_events(event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_events_user_time
    ON usage_events(user_id, occurred_at DESC)
    WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_usage_events_occurred_at
    ON usage_events(occurred_at DESC);

-- -----------------------------------------------------------------------------
-- metering_dead_letter : failed writes, retained for replay + paging
-- -----------------------------------------------------------------------------
-- When a write to `usage_events` fails for any reason (DB unreachable, schema
-- drift, constraint violation), the writer enqueues the failed payload here
-- so the oncall agent can replay it. `notified_at` is set when the on-call
-- has been paged; unset rows are the pageable backlog.

CREATE TABLE IF NOT EXISTS metering_dead_letter (
    id                UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    failed_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    error_class       TEXT         NOT NULL,
    error_message     TEXT         NOT NULL DEFAULT '',
    payload           JSONB        NOT NULL,
    retry_count       INTEGER      NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
    notified_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_dead_letter_failed_at
    ON metering_dead_letter(failed_at DESC);
CREATE INDEX IF NOT EXISTS idx_dead_letter_unnotified
    ON metering_dead_letter(failed_at DESC)
    WHERE notified_at IS NULL;

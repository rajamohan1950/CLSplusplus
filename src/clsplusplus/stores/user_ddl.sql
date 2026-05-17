-- CLS++ User Management Schema
-- Auto-applied on first connection by UserStore._init_schema()

-- gen_random_uuid() requires pgcrypto on PG < 13
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT,
    google_id       TEXT UNIQUE,
    github_id       TEXT UNIQUE,
    name            TEXT NOT NULL DEFAULT '',
    avatar_url      TEXT,
    tier            TEXT NOT NULL DEFAULT 'free'
                    CHECK (tier IN ('free', 'pro', 'business', 'enterprise')),
    is_admin        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Migrations for existing tables (safe on fresh tables too)
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS github_id TEXT UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS tier TEXT NOT NULL DEFAULT 'free';
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN NOT NULL DEFAULT FALSE;

-- Subscription lifecycle (added for PR "subscription expiry").
-- expires_at is NULL on free users and on paid users without a known renewal
-- date (e.g. lifetime deals). The SubscriptionWatchdog only downgrades rows
-- where expires_at IS NOT NULL AND expires_at < now().
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_expires_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN IF NOT EXISTS subscription_status TEXT;  -- trial|active|cancelled|halted|expired
ALTER TABLE users ADD COLUMN IF NOT EXISTS razorpay_subscription_id TEXT;
-- 'trial' marks a brand-new free user inside the 30-day launch-quota window.
-- After subscription_expires_at the free monthly cap drops to the small
-- permanent number (see tiers.effective_monthly_cap); the status flip is cosmetic.
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_subscription_status_check;
ALTER TABLE users ADD CONSTRAINT users_subscription_status_check
    CHECK (subscription_status IS NULL
           OR subscription_status IN ('trial', 'active', 'cancelled', 'halted', 'expired'));
CREATE INDEX IF NOT EXISTS idx_users_subscription_expiry
    ON users(subscription_expires_at)
    WHERE subscription_expires_at IS NOT NULL AND tier != 'free';

-- Drop old CHECK constraint and add new one (safe if doesn't exist)
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_tier_check;
ALTER TABLE users ADD CONSTRAINT users_tier_check CHECK (tier IN ('free', 'pro', 'business', 'enterprise'));

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id) WHERE google_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_github_id ON users(github_id) WHERE github_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_tier ON users(tier);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at DESC);

CREATE TABLE IF NOT EXISTS revenue_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    event_type      TEXT NOT NULL CHECK (event_type IN ('upgrade', 'downgrade', 'cancel')),
    from_tier       TEXT NOT NULL,
    to_tier         TEXT NOT NULL,
    monthly_revenue NUMERIC(10,2) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_revenue_events_user ON revenue_events(user_id);
CREATE INDEX IF NOT EXISTS idx_revenue_events_created ON revenue_events(created_at DESC);

-- Overage billing: pay-as-you-go usage past the tier cap is invoiced as a
-- one-time 'overage' revenue event. `period` (YYYY-MM) + the partial unique
-- index below make the billing job idempotent — re-running it never
-- double-invoices a user for the same month.
ALTER TABLE revenue_events ADD COLUMN IF NOT EXISTS period TEXT;
ALTER TABLE revenue_events ADD COLUMN IF NOT EXISTS razorpay_order_id TEXT;
ALTER TABLE revenue_events ADD COLUMN IF NOT EXISTS amount_cents INTEGER;
ALTER TABLE revenue_events DROP CONSTRAINT IF EXISTS revenue_events_event_type_check;
ALTER TABLE revenue_events ADD CONSTRAINT revenue_events_event_type_check
    CHECK (event_type IN ('upgrade', 'downgrade', 'cancel', 'overage'));
CREATE UNIQUE INDEX IF NOT EXISTS idx_revenue_events_overage_period
    ON revenue_events(user_id, period)
    WHERE event_type = 'overage';

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash  TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reset_tokens_hash ON password_reset_tokens(token_hash);
CREATE INDEX IF NOT EXISTS idx_reset_tokens_user ON password_reset_tokens(user_id);

-- Email verification tokens (OTP + magic link)
CREATE TABLE IF NOT EXISTS email_verification_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    otp_code    TEXT NOT NULL,
    token_hash  TEXT NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_verify_tokens_user ON email_verification_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_verify_tokens_hash ON email_verification_tokens(token_hash);

-- Pending registrations (verify email BEFORE creating user)
CREATE TABLE IF NOT EXISTS pending_registrations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    otp_code        TEXT NOT NULL,
    token_hash      TEXT NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pending_reg_email ON pending_registrations(email);
CREATE INDEX IF NOT EXISTS idx_pending_reg_token ON pending_registrations(token_hash);

-- Revenue events: add CASCADE for user deletion
ALTER TABLE revenue_events DROP CONSTRAINT IF EXISTS revenue_events_user_id_fkey;
ALTER TABLE revenue_events ADD CONSTRAINT revenue_events_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;

-- User feedback / satisfaction (CSAT).
-- One row per explicit feedback submission. `score` is a 1-5 rating;
-- `sentiment` is the coarse thumbs up/down a one-click widget can send
-- without forcing a number. `context` is a free-form tag for where the
-- feedback came from (e.g. 'integrate-page', 'memory-viewer'). CSAT is
-- computed as the share of 4-5 scores; trend is bucketed by day.
CREATE TABLE IF NOT EXISTS user_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    score       SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 5),
    sentiment   TEXT NOT NULL CHECK (sentiment IN ('up', 'down')),
    comment     TEXT,
    context     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_feedback_user ON user_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_user_feedback_created ON user_feedback(created_at DESC);

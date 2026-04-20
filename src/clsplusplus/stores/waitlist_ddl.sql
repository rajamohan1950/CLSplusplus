-- CLS++ Waitlist Schema
-- Auto-applied on first connection by WaitlistStore._init_schema()

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Email OTP cooldown / pending verifications (before they hit the queue)
CREATE TABLE IF NOT EXISTS waitlist_pending_otp (
    email           TEXT PRIMARY KEY,
    otp_code        TEXT NOT NULL,
    source_variant  TEXT,
    expires_at      TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Actual waitlist visitors (verified emails, waiting → invited → activated)
CREATE TABLE IF NOT EXISTS waitlist_visitors (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email              TEXT NOT NULL UNIQUE,
    status             TEXT NOT NULL DEFAULT 'waiting'
                       CHECK (status IN ('waiting', 'invited', 'activated', 'expired', 'cancelled')),
    source_variant     TEXT,
    verified_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    invited_at         TIMESTAMPTZ,
    invite_token_hash  TEXT,
    invite_expires_at  TIMESTAMPTZ,
    activated_at       TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Position the user was last notified about. We email a "you moved up"
-- message every time a user's real position improves by 10 or more. This
-- column holds the most recent notified value so the scheduler can diff.
ALTER TABLE waitlist_visitors
    ADD COLUMN IF NOT EXISTS last_notified_position INTEGER;

CREATE INDEX IF NOT EXISTS idx_waitlist_status_created ON waitlist_visitors(status, created_at);
CREATE INDEX IF NOT EXISTS idx_waitlist_email ON waitlist_visitors(email);
CREATE INDEX IF NOT EXISTS idx_waitlist_invite_hash ON waitlist_visitors(invite_token_hash)
    WHERE invite_token_hash IS NOT NULL;

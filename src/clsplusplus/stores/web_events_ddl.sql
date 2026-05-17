-- CLS++ Website Traffic / Funnel Analytics Schema
-- Auto-applied on first connection by WebEventsStore._init_schema()
--
-- First-party event capture for the marketing site. Powers the admin
-- conversion-funnel dashboard (visitor -> signup -> active user) and the
-- top-pages / engagement rankings. Public POST /v1/events/track writes here.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Raw web events: one row per page view or click. Kept lightweight on
-- purpose -- no joins on the write path, just an append.
CREATE TABLE IF NOT EXISTS web_events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 'pageview' or 'click' (extensible: free text, capped by length).
    event       TEXT NOT NULL,
    -- Path of the page the event happened on, e.g. '/integrate.html'.
    page        TEXT NOT NULL DEFAULT '',
    -- HTTP referrer or marketing source, e.g. 'google', '/'.
    ref         TEXT NOT NULL DEFAULT '',
    -- Anonymous browser session id (client-generated, stable per visit).
    -- Unique-visitor counts and funnel entry are computed from this.
    session_id  TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT web_events_event_len CHECK (char_length(event) BETWEEN 1 AND 64),
    CONSTRAINT web_events_page_len CHECK (char_length(page) <= 512),
    CONSTRAINT web_events_ref_len CHECK (char_length(ref) <= 512),
    CONSTRAINT web_events_session_len CHECK (char_length(session_id) <= 128)
);

CREATE INDEX IF NOT EXISTS idx_web_events_created ON web_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_web_events_session ON web_events(session_id);
CREATE INDEX IF NOT EXISTS idx_web_events_event_page ON web_events(event, page);

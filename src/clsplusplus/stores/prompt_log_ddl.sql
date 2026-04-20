-- CLS++ Prompt Log & Context Log Schema
-- Append-only conversation archive + injection audit trail.
-- These tables are NEVER on the hot recall path — only for persistence and UI.

-- ═══════════════════════════════════════════════════════════════════════════
-- prompt_log: Every prompt from every LLM session, verbatim.
-- Append-only. Rows are never updated (except GDPR erasure).
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS prompt_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    session_id      TEXT NOT NULL,
    sequence_num    INT NOT NULL DEFAULT 0,
    role            TEXT NOT NULL
                    CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    content_hash    TEXT NOT NULL,
    llm_provider    TEXT NOT NULL DEFAULT 'unknown',
    llm_model       TEXT,
    client_type     TEXT NOT NULL DEFAULT 'api',
    namespace       TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Hot query paths
CREATE INDEX IF NOT EXISTS idx_pl_user_ts
    ON prompt_log (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_pl_session
    ON prompt_log (session_id, sequence_num);
CREATE INDEX IF NOT EXISTS idx_pl_ns_ts
    ON prompt_log (namespace, created_at DESC);

-- Idempotent dedup: store hook can re-send same transcript entries
CREATE UNIQUE INDEX IF NOT EXISTS idx_pl_dedup
    ON prompt_log (content_hash, session_id);


-- ═══════════════════════════════════════════════════════════════════════════
-- context_log: What memories were injected into which LLM session.
-- Replaces volatile in-memory dicts in api.py and local_routes.py.
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS context_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL,
    namespace       TEXT NOT NULL,
    session_id      TEXT,
    llm_provider    TEXT NOT NULL DEFAULT 'unknown',
    llm_model       TEXT,
    query           TEXT NOT NULL,
    memories_sent   JSONB NOT NULL DEFAULT '[]'::jsonb,
    memory_ids      JSONB NOT NULL DEFAULT '[]'::jsonb,
    memory_count    INT NOT NULL DEFAULT 0,
    latency_ms      INT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cl_ns_ts
    ON context_log (namespace, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cl_user_ts
    ON context_log (user_id, created_at DESC);


-- ═══════════════════════════════════════════════════════════════════════════
-- namespace_aliases: Maps every auth path to one canonical namespace per user.
-- API key → user-79a40cd1, cookie → user-8786c89f → SAME canonical.
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS namespace_aliases (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id),
    alias           TEXT NOT NULL UNIQUE,
    canonical       TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'manual',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nsa_alias
    ON namespace_aliases (alias);
CREATE INDEX IF NOT EXISTS idx_nsa_canonical
    ON namespace_aliases (canonical);
CREATE INDEX IF NOT EXISTS idx_nsa_user
    ON namespace_aliases (user_id);

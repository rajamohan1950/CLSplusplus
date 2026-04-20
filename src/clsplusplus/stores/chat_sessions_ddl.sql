-- CLS++ Demo Chat Sessions
-- Persists the _ChatSession dataclass used by /v1/chat/sessions/* so users
-- can replay earlier demos from the UI.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS chat_sessions (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID,                              -- NULL for anonymous sessions
    namespace    TEXT NOT NULL,
    name         TEXT NOT NULL DEFAULT 'Untitled',
    messages     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(user_id, updated_at DESC)
    WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chat_sessions_namespace ON chat_sessions(namespace, updated_at DESC);

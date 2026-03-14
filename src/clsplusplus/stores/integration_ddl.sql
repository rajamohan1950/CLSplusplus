-- ============================================================================
-- CLS++ Integration DDL — Self-Service Integration Management
-- ============================================================================
-- 10-year-forward design:
--   • UUID PKs (no sequential ID leaks)
--   • SHA-256 hashed secrets (never plaintext)
--   • Soft-delete with audit trail (revoked_at / deleted_at)
--   • JSONB for schema-free evolution (scopes, metadata, filters)
--   • Timestamptz everywhere
--   • Composite indexes on hot query paths
-- ============================================================================

-- 1. INTEGRATIONS — The "app" entity (registered service/integration)
CREATE TABLE IF NOT EXISTS integrations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    namespace       TEXT NOT NULL DEFAULT 'default',
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'suspended', 'deleted')),

    -- Owner / contact (future: link to user/org table)
    owner_email     TEXT,

    -- Extensible metadata (logo_url, homepage, etc.)
    metadata        JSONB NOT NULL DEFAULT '{}',

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT integrations_name_len CHECK (char_length(name) BETWEEN 1 AND 128),
    CONSTRAINT integrations_namespace_len CHECK (char_length(namespace) BETWEEN 1 AND 64)
);

CREATE INDEX IF NOT EXISTS idx_integrations_namespace ON integrations(namespace);
CREATE INDEX IF NOT EXISTS idx_integrations_status ON integrations(status) WHERE status = 'active';


-- 2. API_CREDENTIALS — Scoped, rotatable API keys
CREATE TABLE IF NOT EXISTS api_credentials (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id  UUID NOT NULL REFERENCES integrations(id),

    -- Key identification
    key_prefix      TEXT NOT NULL,       -- e.g. "cls_live_a1b2" (first 12 chars, safe to display)
    key_hash        TEXT NOT NULL,       -- SHA-256 of full key (for lookup)
    key_hint        TEXT NOT NULL,       -- e.g. "cls_live_****...c3d4" (masked for UI)

    -- Scoping (JSONB array: ["memories:read", "memories:write", "webhooks:manage"])
    scopes          JSONB NOT NULL DEFAULT '["memories:read", "memories:write"]',
    label           TEXT DEFAULT '',     -- human-friendly name

    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'rotated', 'revoked', 'expired')),
    expires_at      TIMESTAMPTZ,         -- NULL = never expires
    last_used_at    TIMESTAMPTZ,
    rotated_from_id UUID,                -- links to predecessor key (rotation chain)
    grace_until     TIMESTAMPTZ,         -- old key still valid until this time after rotation

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ,

    CONSTRAINT api_credentials_prefix_len CHECK (char_length(key_prefix) >= 8)
);

CREATE INDEX IF NOT EXISTS idx_api_credentials_hash ON api_credentials(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_credentials_integration ON api_credentials(integration_id);
CREATE INDEX IF NOT EXISTS idx_api_credentials_active ON api_credentials(status) WHERE status = 'active';


-- 3. WEBHOOK_SUBSCRIPTIONS — Event subscriptions with URL, secret, filters
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id  UUID NOT NULL REFERENCES integrations(id),

    -- Delivery target
    url             TEXT NOT NULL,
    description     TEXT DEFAULT '',

    -- Event filter (JSONB array: ["memory.created", "memory.promoted", "*"])
    events          JSONB NOT NULL DEFAULT '["*"]',

    -- Security: HMAC-SHA256 signing
    secret_hash     TEXT NOT NULL,       -- SHA-256 of webhook secret
    secret_hint     TEXT NOT NULL,       -- masked for UI

    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'paused', 'disabled', 'deleted')),
    failure_count   INT NOT NULL DEFAULT 0,
    max_failures    INT NOT NULL DEFAULT 10,  -- auto-disable after N consecutive failures

    -- Delivery config
    timeout_ms      INT NOT NULL DEFAULT 10000,
    retry_policy    JSONB NOT NULL DEFAULT '{"max_retries": 5, "backoff": "exponential", "initial_delay_ms": 1000}',

    -- Namespace filter (NULL = all namespaces for this integration)
    namespace_filter TEXT,

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at      TIMESTAMPTZ,

    CONSTRAINT webhook_url_len CHECK (char_length(url) BETWEEN 10 AND 2048)
);

CREATE INDEX IF NOT EXISTS idx_webhook_subs_integration ON webhook_subscriptions(integration_id);
CREATE INDEX IF NOT EXISTS idx_webhook_subs_active ON webhook_subscriptions(status) WHERE status = 'active';


-- 4. WEBHOOK_DELIVERIES — Delivery log with status, retry count, response
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES webhook_subscriptions(id),

    -- Event payload
    event_type      TEXT NOT NULL,       -- e.g. "memory.created"
    event_id        UUID NOT NULL,       -- unique event ID for dedup
    payload         JSONB NOT NULL,

    -- Delivery status
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'delivered', 'failed', 'retrying')),
    attempt_count   INT NOT NULL DEFAULT 0,
    max_attempts    INT NOT NULL DEFAULT 6,

    -- Response details
    response_status INT,                 -- HTTP status code
    response_body   TEXT,                -- truncated response (max 4KB)
    response_time_ms INT,                -- delivery latency
    error_message   TEXT,                -- if failed

    -- Timing
    scheduled_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    delivered_at    TIMESTAMPTZ,
    next_retry_at   TIMESTAMPTZ,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_sub ON webhook_deliveries(subscription_id);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_status ON webhook_deliveries(status) WHERE status IN ('pending', 'retrying');
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_event ON webhook_deliveries(event_id);


-- 5. OAUTH_CLIENTS — OAuth2 client credentials (future-proof for marketplace)
CREATE TABLE IF NOT EXISTS oauth_clients (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id  UUID NOT NULL REFERENCES integrations(id),

    -- OAuth2 client credentials
    client_id       TEXT NOT NULL UNIQUE,    -- public identifier
    client_secret_hash TEXT NOT NULL,        -- SHA-256 of secret
    client_secret_hint TEXT NOT NULL,        -- masked for UI

    -- Configuration
    grant_types     JSONB NOT NULL DEFAULT '["client_credentials"]',
    scopes          JSONB NOT NULL DEFAULT '["memories:read", "memories:write"]',
    redirect_uris   JSONB NOT NULL DEFAULT '[]',

    -- Token config
    access_token_ttl_seconds  INT NOT NULL DEFAULT 3600,     -- 1 hour
    refresh_token_ttl_seconds INT NOT NULL DEFAULT 2592000,   -- 30 days

    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'suspended', 'revoked')),

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_oauth_clients_client_id ON oauth_clients(client_id);
CREATE INDEX IF NOT EXISTS idx_oauth_clients_integration ON oauth_clients(integration_id);


-- 6. INTEGRATION_EVENTS — Immutable audit log of all integration activity
CREATE TABLE IF NOT EXISTS integration_events (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    integration_id  UUID NOT NULL REFERENCES integrations(id),

    -- Event details
    event_type      TEXT NOT NULL,       -- e.g. "key.created", "webhook.subscribed", "key.rotated"
    actor           TEXT NOT NULL DEFAULT 'system',  -- who triggered (api_key prefix, user email, "system")
    description     TEXT NOT NULL,

    -- Context
    resource_type   TEXT,                -- "api_key", "webhook", "oauth_client"
    resource_id     UUID,                -- ID of the affected resource
    metadata        JSONB NOT NULL DEFAULT '{}',  -- extra context (old_key_prefix, new_scopes, etc.)

    -- Immutable
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_integration_events_integration ON integration_events(integration_id);
CREATE INDEX IF NOT EXISTS idx_integration_events_type ON integration_events(event_type);
CREATE INDEX IF NOT EXISTS idx_integration_events_created ON integration_events(created_at DESC);


-- 7. CONNECTOR_TEMPLATES — Pre-built connector configs (Zapier, LangChain, etc.)
CREATE TABLE IF NOT EXISTS connector_templates (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug            TEXT NOT NULL UNIQUE,    -- e.g. "langchain", "zapier", "n8n"
    name            TEXT NOT NULL,           -- display name
    description     TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'general'
                    CHECK (category IN ('ai_framework', 'automation', 'cloud', 'custom')),

    -- Integration config (declarative: what endpoints to call, what to map)
    config_schema   JSONB NOT NULL DEFAULT '{}',  -- JSON Schema for user config
    default_config  JSONB NOT NULL DEFAULT '{}',  -- sensible defaults
    setup_steps     JSONB NOT NULL DEFAULT '[]',  -- ordered list of setup instructions

    -- Display
    icon_url        TEXT,
    docs_url        TEXT,
    is_featured     BOOLEAN NOT NULL DEFAULT FALSE,

    -- Versioning
    version         TEXT NOT NULL DEFAULT '1.0.0',
    min_api_version TEXT NOT NULL DEFAULT 'v1',

    -- Audit
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_connector_templates_slug ON connector_templates(slug);
CREATE INDEX IF NOT EXISTS idx_connector_templates_category ON connector_templates(category);

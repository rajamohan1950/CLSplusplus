-- CLS++ Role-Based Access Control Schema
-- Auto-applied on first connection by UserStore._init_schema()

-- 1. ROLES — Named bundles of scopes
CREATE TABLE IF NOT EXISTS roles (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    scopes      JSONB NOT NULL DEFAULT '[]',
    is_system   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. GROUPS — Collections of users
CREATE TABLE IF NOT EXISTS groups (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. GROUP_ROLES — Which roles are assigned to which groups (M:N)
CREATE TABLE IF NOT EXISTS group_roles (
    group_id    UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_id, role_id)
);

-- 4. USER_GROUPS — Which users belong to which groups (M:N)
CREATE TABLE IF NOT EXISTS user_groups (
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id    UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, group_id)
);

-- 5. USER_ROLES — Direct role assignment to users (bypasses groups)
CREATE TABLE IF NOT EXISTS user_roles (
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id     UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

-- 6. USER_PERMISSIONS — Direct per-user scope overrides (grant or deny)
CREATE TABLE IF NOT EXISTS user_permissions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope       TEXT NOT NULL,
    granted     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, scope)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_user_groups_user ON user_groups(user_id);
CREATE INDEX IF NOT EXISTS idx_user_groups_group ON user_groups(group_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_user ON user_roles(user_id);
CREATE INDEX IF NOT EXISTS idx_group_roles_group ON group_roles(group_id);
CREATE INDEX IF NOT EXISTS idx_user_permissions_user ON user_permissions(user_id);

-- Seed default roles (idempotent)
INSERT INTO roles (name, description, scopes, is_system) VALUES
    ('viewer', 'Read-only access to memories and usage',
     '["memories:read", "usage:read", "chat:use", "page:chat", "page:docs", "page:memory", "page:dashboard"]', TRUE),
    ('editor', 'Read and write access to memories',
     '["memories:read", "memories:write", "memories:delete", "consolidate", "usage:read", "chat:use", "page:chat", "page:docs", "page:integrate", "page:getting-started", "page:dashboard", "page:memory"]', TRUE),
    ('admin', 'Full access to all features and pages',
     '["memories:read", "memories:write", "memories:delete", "consolidate", "webhooks:manage", "integrations:manage", "usage:read", "admin", "chat:use", "user:upgrade", "page:chat", "page:docs", "page:integrate", "page:getting-started", "page:dashboard", "page:memory"]', TRUE),
    ('super_admin', 'Unrestricted system access',
     '["memories:read", "memories:write", "memories:delete", "consolidate", "webhooks:manage", "integrations:manage", "usage:read", "admin", "chat:use", "user:upgrade", "page:chat", "page:docs", "page:integrate", "page:getting-started", "page:dashboard", "page:memory"]', TRUE)
ON CONFLICT (name) DO NOTHING;

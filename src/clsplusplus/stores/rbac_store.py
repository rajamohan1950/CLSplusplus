"""RBAC Store — PostgreSQL-backed role, group, and permission management."""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


def _row_to_dict(row: asyncpg.Record) -> dict:
    d = dict(row)
    for k, v in d.items():
        if hasattr(v, "hex") and hasattr(v, "int"):
            d[k] = str(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


class RBACStore:
    """PostgreSQL-backed store for RBAC tables."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        _parse_db_url(self.settings.database_url),
                        min_size=1, max_size=5, command_timeout=60,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        ddl_path = os.path.join(os.path.dirname(__file__), "rbac_ddl.sql")
        with open(ddl_path) as f:
            ddl = f.read()
        await conn.execute(ddl)

    # =========================================================================
    # Roles CRUD
    # =========================================================================

    async def list_roles(self) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM roles ORDER BY name")
            return [_row_to_dict(r) for r in rows]

    async def get_role(self, role_id: str) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM roles WHERE id = $1", role_id)
            return _row_to_dict(row) if row else None

    async def create_role(self, name: str, description: str, scopes: list[str]) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO roles (name, description, scopes) VALUES ($1, $2, $3) RETURNING *",
                name, description, json.dumps(scopes),
            )
            return _row_to_dict(row)

    async def update_role(self, role_id: str, description: str = None, scopes: list[str] = None) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Don't allow editing system roles' scopes
            role = await conn.fetchrow("SELECT is_system FROM roles WHERE id = $1", role_id)
            if not role:
                return False
            sets = ["updated_at = NOW()"]
            args = []
            idx = 1
            if description is not None:
                idx += 1
                sets.append(f"description = ${idx}")
                args.append(description)
            if scopes is not None:
                idx += 1
                sets.append(f"scopes = ${idx}")
                args.append(json.dumps(scopes))
            result = await conn.execute(
                f"UPDATE roles SET {', '.join(sets)} WHERE id = $1",
                role_id, *args,
            )
            return result == "UPDATE 1"

    async def delete_role(self, role_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            role = await conn.fetchrow("SELECT is_system FROM roles WHERE id = $1", role_id)
            if not role or role["is_system"]:
                return False
            result = await conn.execute("DELETE FROM roles WHERE id = $1", role_id)
            return result == "DELETE 1"

    # =========================================================================
    # Groups CRUD
    # =========================================================================

    async def list_groups(self) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT g.*, COUNT(ug.user_id) AS member_count
                FROM groups g LEFT JOIN user_groups ug ON ug.group_id = g.id
                GROUP BY g.id ORDER BY g.name
            """)
            return [_row_to_dict(r) for r in rows]

    async def create_group(self, name: str, description: str = "") -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO groups (name, description) VALUES ($1, $2) RETURNING *",
                name, description,
            )
            return _row_to_dict(row)

    async def update_group(self, group_id: str, name: str = None, description: str = None) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            sets = ["updated_at = NOW()"]
            args = []
            idx = 1
            if name is not None:
                idx += 1; sets.append(f"name = ${idx}"); args.append(name)
            if description is not None:
                idx += 1; sets.append(f"description = ${idx}"); args.append(description)
            result = await conn.execute(f"UPDATE groups SET {', '.join(sets)} WHERE id = $1", group_id, *args)
            return result == "UPDATE 1"

    async def delete_group(self, group_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM groups WHERE id = $1", group_id)
            return result == "DELETE 1"

    # =========================================================================
    # Group Roles
    # =========================================================================

    async def get_group_roles(self, group_id: str) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT r.* FROM group_roles gr JOIN roles r ON r.id = gr.role_id WHERE gr.group_id = $1",
                group_id,
            )
            return [_row_to_dict(r) for r in rows]

    async def add_group_role(self, group_id: str, role_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO group_roles (group_id, role_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    group_id, role_id,
                )
                return True
            except Exception:
                return False

    async def remove_group_role(self, group_id: str, role_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM group_roles WHERE group_id = $1 AND role_id = $2", group_id, role_id)
            return result == "DELETE 1"

    # =========================================================================
    # Group Members
    # =========================================================================

    async def get_group_members(self, group_id: str) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT u.id, u.email, u.name, u.tier, u.is_admin FROM user_groups ug JOIN users u ON u.id = ug.user_id WHERE ug.group_id = $1 ORDER BY u.email",
                group_id,
            )
            return [_row_to_dict(r) for r in rows]

    async def add_group_member(self, group_id: str, user_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO user_groups (user_id, group_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    user_id, group_id,
                )
                return True
            except Exception:
                return False

    async def remove_group_member(self, group_id: str, user_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM user_groups WHERE group_id = $1 AND user_id = $2", group_id, user_id)
            return result == "DELETE 1"

    # =========================================================================
    # User Roles (direct assignment)
    # =========================================================================

    async def get_user_roles(self, user_id: str) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT r.* FROM user_roles ur JOIN roles r ON r.id = ur.role_id WHERE ur.user_id = $1",
                user_id,
            )
            return [_row_to_dict(r) for r in rows]

    async def add_user_role(self, user_id: str, role_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(
                    "INSERT INTO user_roles (user_id, role_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    user_id, role_id,
                )
                return True
            except Exception:
                return False

    async def remove_user_role(self, user_id: str, role_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM user_roles WHERE user_id = $1 AND role_id = $2", user_id, role_id)
            return result == "DELETE 1"

    # =========================================================================
    # User Permissions (direct overrides)
    # =========================================================================

    async def get_user_permissions(self, user_id: str) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM user_permissions WHERE user_id = $1 ORDER BY scope",
                user_id,
            )
            return [_row_to_dict(r) for r in rows]

    async def set_user_permission(self, user_id: str, scope: str, granted: bool) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO user_permissions (user_id, scope, granted) VALUES ($1, $2, $3)
                ON CONFLICT (user_id, scope) DO UPDATE SET granted = $3
                RETURNING *
            """, user_id, scope, granted)
            return _row_to_dict(row)

    async def remove_user_permission(self, user_id: str, permission_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM user_permissions WHERE id = $1 AND user_id = $2",
                permission_id, user_id,
            )
            return result == "DELETE 1"

    # =========================================================================
    # Effective Scopes — the key query
    # =========================================================================

    async def get_effective_scopes(self, user_id: str) -> set[str]:
        """Compute effective scopes for a user from roles + groups + overrides."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                WITH role_scopes AS (
                    -- Direct user roles
                    SELECT r.scopes FROM user_roles ur
                    JOIN roles r ON r.id = ur.role_id WHERE ur.user_id = $1
                    UNION ALL
                    -- Group roles
                    SELECT r.scopes FROM user_groups ug
                    JOIN group_roles gr ON gr.group_id = ug.group_id
                    JOIN roles r ON r.id = gr.role_id WHERE ug.user_id = $1
                ),
                all_scopes AS (
                    SELECT DISTINCT jsonb_array_elements_text(scopes) AS scope
                    FROM role_scopes
                )
                -- Base scopes minus denied overrides
                SELECT scope, TRUE AS granted FROM all_scopes
                WHERE scope NOT IN (
                    SELECT scope FROM user_permissions WHERE user_id = $1 AND granted = FALSE
                )
                UNION
                -- Plus granted overrides
                SELECT scope, TRUE AS granted FROM user_permissions
                WHERE user_id = $1 AND granted = TRUE
            """, user_id)
            return {r["scope"] for r in rows}

    async def get_user_groups(self, user_id: str) -> list[dict]:
        """Get groups a user belongs to."""
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT g.* FROM user_groups ug JOIN groups g ON g.id = ug.group_id WHERE ug.user_id = $1 ORDER BY g.name",
                user_id,
            )
            return [_row_to_dict(r) for r in rows]

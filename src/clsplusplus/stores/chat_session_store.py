"""Persistence for demo chat sessions.

The legacy /v1/chat/sessions handlers in api.py stored sessions in a
module-level dict that evaporated on restart. This store backs them with
Postgres so an authenticated user can replay an earlier demo.

Public API (async):
  create(user_id, namespace, name) -> dict
  add_message(session_id, role, content, llm=None) -> dict
  get(session_id, user_id=None) -> dict | None
  list_for_user(user_id, limit=50) -> list[dict]
  delete(session_id, user_id) -> bool
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)


class ChatSessionStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._pool: Optional[asyncpg.Pool] = None
        self._init_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            async with self._init_lock:
                if self._pool is None:
                    dsn = self.settings.database_url
                    if not dsn:
                        raise RuntimeError("CLS_DATABASE_URL not configured")
                    self._pool = await asyncpg.create_pool(dsn, min_size=1, max_size=3)
                    async with self._pool.acquire() as conn:
                        ddl_path = os.path.join(os.path.dirname(__file__), "chat_sessions_ddl.sql")
                        with open(ddl_path) as f:
                            await conn.execute(f.read())
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create(self, user_id: Optional[str], namespace: str, name: str) -> dict:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO chat_sessions (user_id, namespace, name)
                VALUES ($1::uuid, $2, $3)
                RETURNING id, user_id, namespace, name, messages, created_at, updated_at
                """,
                user_id,
                namespace,
                name or "Untitled",
            )
        return _row_to_dict(row)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        llm: Optional[str] = None,
    ) -> Optional[dict]:
        message = {"role": role, "content": content}
        if llm:
            message["llm"] = llm
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE chat_sessions
                SET messages = messages || $1::jsonb,
                    updated_at = NOW()
                WHERE id = $2::uuid
                RETURNING id, user_id, namespace, name, messages, created_at, updated_at
                """,
                json.dumps([message]),
                session_id,
            )
        return _row_to_dict(row) if row else None

    async def get(self, session_id: str, user_id: Optional[str] = None) -> Optional[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            if user_id is not None:
                row = await conn.fetchrow(
                    """
                    SELECT id, user_id, namespace, name, messages, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = $1::uuid AND (user_id = $2::uuid OR user_id IS NULL)
                    """,
                    session_id,
                    user_id,
                )
            else:
                row = await conn.fetchrow(
                    """
                    SELECT id, user_id, namespace, name, messages, created_at, updated_at
                    FROM chat_sessions WHERE id = $1::uuid
                    """,
                    session_id,
                )
        return _row_to_dict(row) if row else None

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, user_id, namespace, name, messages, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = $1::uuid
                ORDER BY updated_at DESC
                LIMIT $2
                """,
                user_id,
                int(limit),
            )
        return [_row_to_dict(r) for r in rows]

    async def delete(self, session_id: str, user_id: str) -> bool:
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM chat_sessions WHERE id = $1::uuid AND user_id = $2::uuid",
                session_id,
                user_id,
            )
        # asyncpg returns "DELETE N"
        try:
            n = int(result.rsplit(" ", 1)[-1])
        except Exception:
            n = 0
        return n > 0


def _row_to_dict(row) -> dict:
    messages = row["messages"]
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except Exception:
            messages = []
    return {
        "id": str(row["id"]),
        "user_id": str(row["user_id"]) if row["user_id"] else None,
        "namespace": row["namespace"],
        "name": row["name"],
        "messages": messages or [],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }

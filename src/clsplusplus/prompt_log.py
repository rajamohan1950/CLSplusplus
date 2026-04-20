"""
CLS++ Prompt Log Store — Append-Only Conversation Archive

Fire-and-forget writes to PostgreSQL. NEVER on the hot recall path.
Used for:
  - Memory Viewer conversation history
  - Context injection audit trail
  - Analytics and billing
  - GDPR/data export

All writes use asyncio.create_task() — they never block the LLM response.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import asyncpg

from clsplusplus.config import Settings

logger = logging.getLogger(__name__)

_DDL_PATH = Path(__file__).parent / "stores" / "prompt_log_ddl.sql"


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


class PromptLogStore:
    """Append-only prompt log. Fire-and-forget writes, indexed reads."""

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
                        min_size=1,
                        max_size=5,
                        command_timeout=30,
                    )
                    async with self._pool.acquire() as conn:
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        """Create tables if not exist."""
        if _DDL_PATH.exists():
            ddl = _DDL_PATH.read_text()
            # Execute each statement separately (asyncpg doesn't support multi-statement)
            for stmt in ddl.split(";"):
                stmt = stmt.strip()
                if stmt and not stmt.startswith("--"):
                    try:
                        await conn.execute(stmt)
                    except Exception as e:
                        # Non-fatal: table may already exist, index may conflict
                        logger.debug("DDL statement skipped: %s", e)

    # ─── Write ────────────────────────────────────────────────────────

    async def append(self, user_id: str, session_id: str, sequence_num: int,
                     role: str, content: str, llm_provider: str,
                     llm_model: str, client_type: str, namespace: str,
                     metadata: Optional[dict] = None) -> None:
        """Append a single prompt. Idempotent via content_hash + session_id."""
        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        pool = await self.get_pool()
        try:
            await pool.execute("""
                INSERT INTO prompt_log
                    (user_id, session_id, sequence_num, role, content,
                     content_hash, llm_provider, llm_model, client_type,
                     namespace, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (content_hash, session_id) DO NOTHING
            """, user_id, session_id, sequence_num, role, content,
                 content_hash, llm_provider, llm_model or "", client_type,
                 namespace, json.dumps(metadata or {}))
        except Exception as e:
            logger.warning("prompt_log append failed (non-fatal): %s", e)

    async def batch_append(self, user_id: str, namespace: str,
                           session_id: str, llm_provider: str,
                           llm_model: str, client_type: str,
                           entries: list[dict]) -> int:
        """Batch append multiple prompts. Returns count inserted."""
        pool = await self.get_pool()
        inserted = 0
        # Use a single connection for the batch
        async with pool.acquire() as conn:
            for entry in entries:
                content = entry.get("content", "")
                if not content:
                    continue
                content_hash = hashlib.sha256(
                    content.encode("utf-8", errors="replace")).hexdigest()
                try:
                    result = await conn.execute("""
                        INSERT INTO prompt_log
                            (user_id, session_id, sequence_num, role, content,
                             content_hash, llm_provider, llm_model, client_type,
                             namespace, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (content_hash, session_id) DO NOTHING
                    """, user_id, session_id,
                         entry.get("sequence_num", 0),
                         entry.get("role", "user"),
                         content,
                         content_hash, llm_provider, llm_model or "",
                         client_type, namespace,
                         json.dumps(entry.get("metadata", {})))
                    if "INSERT" in result:
                        inserted += 1
                except Exception as e:
                    logger.debug("batch entry skipped: %s", e)
        return inserted

    # ─── Read ─────────────────────────────────────────────────────────

    async def get_session(self, session_id: str,
                          limit: int = 100) -> list[dict]:
        """Get conversation history for a session, ordered by sequence."""
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT id, user_id, session_id, sequence_num, role, content,
                   llm_provider, llm_model, namespace, metadata, created_at
            FROM prompt_log
            WHERE session_id = $1
            ORDER BY sequence_num ASC
            LIMIT $2
        """, session_id, limit)
        return [dict(r) for r in rows]

    async def get_user_sessions(self, user_id: str,
                                limit: int = 20) -> list[dict]:
        """List distinct sessions for a user, most recent first."""
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT session_id, llm_provider, llm_model,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_at,
                   COUNT(*) AS message_count
            FROM prompt_log
            WHERE user_id = $1
            GROUP BY session_id, llm_provider, llm_model
            ORDER BY MAX(created_at) DESC
            LIMIT $2
        """, user_id, limit)
        return [dict(r) for r in rows]

    async def get_user_sessions_by_namespace(self, namespace: str,
                                              limit: int = 20) -> list[dict]:
        """List sessions by namespace (for Memory Viewer)."""
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT session_id, llm_provider, llm_model,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_at,
                   COUNT(*) AS message_count
            FROM prompt_log
            WHERE namespace = $1
            GROUP BY session_id, llm_provider, llm_model
            ORDER BY MAX(created_at) DESC
            LIMIT $2
        """, namespace, limit)
        return [dict(r) for r in rows]

    async def get_timeline(self, namespace: str, limit: int = 50,
                           before: Optional[datetime] = None) -> list[dict]:
        """Paginated timeline of all prompts across sessions."""
        pool = await self.get_pool()
        if before:
            rows = await pool.fetch("""
                SELECT id, session_id, sequence_num, role, content,
                       llm_provider, llm_model, created_at
                FROM prompt_log
                WHERE namespace = $1 AND created_at < $2
                ORDER BY created_at DESC
                LIMIT $3
            """, namespace, before, limit)
        else:
            rows = await pool.fetch("""
                SELECT id, session_id, sequence_num, role, content,
                       llm_provider, llm_model, created_at
                FROM prompt_log
                WHERE namespace = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, namespace, limit)
        return [dict(r) for r in rows]


class ContextLogStore:
    """Persistent context injection log. Replaces volatile in-memory dicts."""

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
                        min_size=1,
                        max_size=5,
                        command_timeout=30,
                    )
                    # Schema created by PromptLogStore._init_schema (same DDL file)
        return self._pool

    async def append(self, user_id: str, namespace: str,
                     session_id: str, llm_provider: str,
                     query: str, memories_sent: list[str],
                     memory_ids: list[str], memory_count: int,
                     latency_ms: Optional[int] = None,
                     llm_model: Optional[str] = None) -> None:
        """Fire-and-forget: log a context injection event."""
        pool = await self.get_pool()
        try:
            await pool.execute("""
                INSERT INTO context_log
                    (user_id, namespace, session_id, llm_provider, llm_model,
                     query, memories_sent, memory_ids, memory_count, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, user_id, namespace, session_id or "",
                 llm_provider, llm_model or "",
                 query[:500],
                 json.dumps(memories_sent[:20]),
                 json.dumps(memory_ids[:20]),
                 memory_count,
                 latency_ms)
        except Exception as e:
            logger.warning("context_log append failed (non-fatal): %s", e)

    async def get_by_namespace(self, namespace: str,
                               limit: int = 50) -> list[dict]:
        """Get context injection history for a namespace."""
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT id, session_id, llm_provider, llm_model, query,
                   memories_sent, memory_ids, memory_count, latency_ms, created_at
            FROM context_log
            WHERE namespace = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, namespace, limit)
        return [dict(r) for r in rows]

    async def get_by_user(self, user_id: str,
                          limit: int = 50) -> list[dict]:
        """Cross-namespace context log for unified view."""
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT id, namespace, session_id, llm_provider, llm_model, query,
                   memories_sent, memory_ids, memory_count, latency_ms, created_at
            FROM context_log
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, user_id, limit)
        return [dict(r) for r in rows]

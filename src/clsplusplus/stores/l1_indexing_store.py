"""L1 Indexing Store - Hippocampus equivalent.

Fast episodic encoding, pattern completion. pgvector + metadata.
"""

import asyncio
import json
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores.base import BaseStore
from clsplusplus.tracer import tracer

logger = logging.getLogger(__name__)

# Allowlist of columns that can be updated via update_scores (prevents SQL injection)
_ALLOWED_SCORE_COLUMNS = frozenset({
    "confidence", "salience", "usage_count", "authority",
    "conflict_score", "surprise", "promotion_score",
})


def _parse_db_url(url: str) -> str:
    """Convert postgresql:// to postgres:// for asyncpg if needed."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


class L1IndexingStore(BaseStore):
    """L1: Indexing Store - episodic kNN retrieval."""

    level = StoreLevel.L1

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None
        self._pool_lock = asyncio.Lock()

    async def get_pool(self) -> asyncpg.Pool:
        """Thread-safe lazy pool initialization with double-checked locking."""
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    self._pool = await asyncpg.create_pool(
                        _parse_db_url(self.settings.database_url),
                        min_size=1,
                        max_size=10,
                        command_timeout=60,
                    )
                    # Register pgvector (optional — falls back to FLOAT8[] if not installed)
                    async with self._pool.acquire() as conn:
                        try:
                            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                            await register_vector(conn)
                            self._has_pgvector = True
                        except Exception:
                            self._has_pgvector = False
                            logger.info("pgvector not available — using FLOAT8[] for embeddings")
                        await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        """Create L1 table if not exists."""
        emb_type = "vector(384)" if getattr(self, '_has_pgvector', False) else "FLOAT8[]"
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS l1_memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                store_level TEXT DEFAULT 'L1',
                source TEXT DEFAULT 'user',
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                confidence REAL DEFAULT 0.5,
                version INT DEFAULT 1,
                checksum TEXT,
                lineage JSONB DEFAULT '[]',
                salience REAL DEFAULT 0.5,
                usage_count INT DEFAULT 0,
                authority REAL DEFAULT 0.5,
                conflict_score REAL DEFAULT 0.0,
                surprise REAL DEFAULT 0.0,
                promotion_score REAL DEFAULT 0.0,
                metadata JSONB DEFAULT '{{}}'::jsonb,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                embedding {emb_type},
                event_at TIMESTAMPTZ DEFAULT NULL,
                superseded BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        # Idempotent migrations for existing tables (safe on fresh tables too)
        await conn.execute("""
            ALTER TABLE l1_memories ADD COLUMN IF NOT EXISTS event_at TIMESTAMPTZ DEFAULT NULL
        """)
        await conn.execute("""
            ALTER TABLE l1_memories ADD COLUMN IF NOT EXISTS superseded BOOLEAN DEFAULT FALSE
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_l1_namespace ON l1_memories(namespace)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_l1_event_at ON l1_memories(namespace, event_at)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_l1_superseded ON l1_memories(namespace, superseded)
        """)
        # IVFFlat index - requires rows; create when table has data
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_l1_embedding ON l1_memories
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
            """)
        except Exception:
            pass  # Index may fail on empty table

    async def write(
        self,
        item: MemoryItem,
        trace_id: Optional[str] = None,
        parent_hop_id: Optional[str] = None,
    ) -> MemoryItem:
        """Write to L1 with embedding."""
        item.store_level = StoreLevel.L1
        pool = await self.get_pool()
        embedding_val = None
        if item.embedding:
            if getattr(self, '_has_pgvector', False):
                embedding_val = "[" + ",".join(str(x) for x in item.embedding) + "]"
            else:
                embedding_val = list(item.embedding)

        _span_ctx = (
            tracer.span(trace_id, "sql.insert", "postgres",
                        _parent=parent_hop_id,
                        table="l1_memories", op="upsert", item_id=item.id)
            if trace_id else nullcontext()
        )
        with _span_ctx:
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO l1_memories (
                        id, namespace, text, store_level, source, timestamp, confidence,
                        version, checksum, lineage, salience, usage_count, authority,
                        conflict_score, surprise, promotion_score, metadata,
                        subject, predicate, object, embedding, event_at, superseded
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        confidence = EXCLUDED.confidence,
                        version = l1_memories.version + 1,
                        salience = EXCLUDED.salience,
                        usage_count = l1_memories.usage_count + 1,
                        promotion_score = EXCLUDED.promotion_score,
                        embedding = EXCLUDED.embedding,
                        event_at = COALESCE(EXCLUDED.event_at, l1_memories.event_at),
                        superseded = EXCLUDED.superseded
                """,
                    item.id, item.namespace, item.text, item.store_level.value,
                    item.source, item.timestamp, item.confidence, item.version,
                    item.checksum, json.dumps(item.lineage), item.salience,
                    item.usage_count, item.authority, item.conflict_score,
                    item.surprise, item.promotion_score, json.dumps(item.metadata),
                    item.subject, item.predicate, item.object, embedding_val,
                    item.event_at, item.superseded,
                )
        return item

    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        """kNN search by embedding similarity."""
        pool = await self.get_pool()

        if getattr(self, '_has_pgvector', False):
            emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            rows = await pool.fetch("""
                SELECT id, namespace, text, store_level, source, timestamp, confidence,
                       version, checksum, lineage, salience, usage_count, authority,
                       conflict_score, surprise, promotion_score, metadata,
                       subject, predicate, object, embedding
                FROM l1_memories
                WHERE namespace = $1 AND confidence >= $2 AND embedding IS NOT NULL
                ORDER BY embedding <=> $3::vector
                LIMIT $4
            """, namespace, min_confidence, emb_str, limit)
        else:
            # Without pgvector: fetch all and sort by cosine in Python
            rows = await pool.fetch("""
                SELECT id, namespace, text, store_level, source, timestamp, confidence,
                       version, checksum, lineage, salience, usage_count, authority,
                       conflict_score, surprise, promotion_score, metadata,
                       subject, predicate, object, embedding
                FROM l1_memories
                WHERE namespace = $1 AND confidence >= $2 AND embedding IS NOT NULL
                LIMIT 1000
            """, namespace, min_confidence)
            # Sort by cosine similarity in Python
            import numpy as np
            q = np.array(query_embedding)
            def _cos(row):
                e = row['embedding']
                if not e: return -1
                v = np.array(e)
                return float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            rows = sorted(rows, key=_cos, reverse=True)[:limit]

        return [self._row_to_item(r) for r in rows]

    def _row_to_item(self, row) -> MemoryItem:
        """Convert DB row to MemoryItem."""
        emb = None
        if row["embedding"]:
            emb = [float(x) for x in row["embedding"]]
        # event_at column may be absent from old rows before migration
        event_at = None
        try:
            event_at = row["event_at"]
        except (KeyError, Exception):
            pass
        superseded = False
        try:
            superseded = bool(row["superseded"])
        except (KeyError, Exception):
            pass
        return MemoryItem(
            id=row["id"],
            namespace=row["namespace"],
            text=row["text"],
            store_level=StoreLevel(row["store_level"]),
            source=row["source"],
            timestamp=row["timestamp"],
            confidence=float(row["confidence"]),
            version=row["version"],
            checksum=row["checksum"],
            lineage=json.loads(row["lineage"]) if isinstance(row["lineage"], str) else (row["lineage"] or []),
            salience=float(row["salience"]),
            usage_count=row["usage_count"],
            authority=float(row["authority"]),
            conflict_score=float(row["conflict_score"]),
            surprise=float(row["surprise"]),
            promotion_score=float(row["promotion_score"]),
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else (row["metadata"] or {}),
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            embedding=emb,
            event_at=event_at,
            superseded=superseded,
        )

    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        """Get by ID."""
        pool = await self.get_pool()
        row = await pool.fetchrow(
            "SELECT * FROM l1_memories WHERE id = $1 AND namespace = $2",
            item_id, namespace,
        )
        if row:
            return self._row_to_item(row)
        return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        """Delete from L1."""
        pool = await self.get_pool()
        result = await pool.execute(
            "DELETE FROM l1_memories WHERE id = $1 AND namespace = $2",
            item_id, namespace,
        )
        return "DELETE 1" in result

    async def list_for_sleep(self, namespace: str, limit: int = 50000) -> list[MemoryItem]:
        """List items for sleep cycle / init preload.

        Default limit raised to 50,000 so a namespace with a large number of
        memories is fully warm on first use.  For namespaces approaching 1M
        items the PhaseMemoryEngine keeps the 1,000 hottest items in memory
        while L1 vector search handles the rest via kNN.
        """
        pool = await self.get_pool()
        rows = await pool.fetch("""
            SELECT * FROM l1_memories
            WHERE namespace = $1 AND (superseded IS NULL OR superseded = FALSE)
            ORDER BY confidence DESC, timestamp DESC
            LIMIT $2
        """, namespace, limit)
        return [self._row_to_item(r) for r in rows]

    async def list_namespaces(self) -> list[str]:
        """Return every distinct namespace that has at least one item in L1.

        Used at startup to discover which namespaces to preload into the
        PhaseMemoryEngine so the first user request for any namespace is instant.
        """
        pool = await self.get_pool()
        rows = await pool.fetch(
            "SELECT DISTINCT namespace FROM l1_memories ORDER BY namespace"
        )
        return [r["namespace"] for r in rows]

    async def count(self, namespace: str) -> int:
        """Return the total number of items stored for a namespace."""
        pool = await self.get_pool()
        return await pool.fetchval(
            "SELECT COUNT(*) FROM l1_memories WHERE namespace = $1", namespace
        )

    async def ensure_vector_index_scale(self) -> None:
        """Re-tune the IVFFlat index lists based on actual row count.

        Rule of thumb: lists ≈ sqrt(total_rows).
          •  <  10k rows  →  lists =  100  (default, no rebuild needed)
          •  < 100k rows  →  lists =  316
          •  <   1M rows  →  lists = 1000
          •  >= 1M rows   →  lists = 2000

        CONCURRENTLY index operations must each run on their own connection
        outside any transaction block — asyncpg autocommit satisfies this as
        long as we do NOT share a connection across the DROP and CREATE.
        Called once at startup — a no-op when the row count hasn't crossed a
        threshold boundary since last build.
        """
        pool = await self.get_pool()

        # Step 1: count rows (own connection, fast)
        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM l1_memories") or 0

        if total < 1000:
            return  # Too few rows for IVFFlat to matter

        lists = 100
        if total >= 1_000_000:
            lists = 2000
        elif total >= 100_000:
            lists = 1000
        elif total >= 10_000:
            lists = 316

        try:
            # Step 2: DROP CONCURRENTLY — own connection (no shared transaction)
            async with pool.acquire() as conn:
                await conn.execute(
                    "DROP INDEX CONCURRENTLY IF EXISTS idx_l1_embedding"
                )

            # Step 3: CREATE CONCURRENTLY — own connection (no shared transaction)
            async with pool.acquire() as conn:
                await conn.execute(
                    f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_l1_embedding "
                    f"ON l1_memories USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = {lists})"
                )

            logger.info(
                "IVFFlat index rebuilt CONCURRENTLY: total_rows=%d lists=%d",
                total, lists,
            )
        except Exception as e:
            logger.warning("IVFFlat scale rebuild failed (non-fatal): %s", e)

    async def update_scores(self, item_id: str, namespace: str, **kwargs) -> bool:
        """Update plasticity scores. Column names are validated against an allowlist."""
        pool = await self.get_pool()
        updates = []
        values = []
        i = 1
        for k, v in kwargs.items():
            if k not in _ALLOWED_SCORE_COLUMNS:
                raise ValueError(f"Column '{k}' not in allowed update columns: {_ALLOWED_SCORE_COLUMNS}")
            updates.append(f"{k} = ${i}")
            values.append(v)
            i += 1
        if not updates:
            return False
        values.extend([item_id, namespace])
        await pool.execute(
            f"UPDATE l1_memories SET {', '.join(updates)} WHERE id = ${i} AND namespace = ${i+1}",
            *values,
        )
        return True

    async def update_superseded(self, item_id: str, namespace: str) -> None:
        """Mark an item as superseded (a newer fact on the same topic now exists).

        Superseded items are hidden from default reads but never physically deleted.
        Callers can retrieve them by passing ``include_superseded=True`` on a read.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE l1_memories SET superseded = TRUE WHERE id = $1 AND namespace = $2",
                item_id, namespace,
            )

    async def close(self) -> None:
        """Cleanly shut down the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def health(self) -> dict:
        """Health check."""
        try:
            pool = await self.get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy", "store": "L1"}
        except Exception as e:
            logger.error("L1 health check failed: %s", e)
            return {"status": "unhealthy", "store": "L1", "error": "Connection failed"}

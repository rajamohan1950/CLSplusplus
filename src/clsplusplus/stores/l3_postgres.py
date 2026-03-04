"""L3 Deep Recess - PostgreSQL backend (free tier, no MinIO).

Stable long-term archive stored in Postgres. Use when MinIO is not available.
"""

import json
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores.base import BaseStore


def _parse_db_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgres://", 1)
    return url


class L3PostgresStore(BaseStore):
    """L3: Deep Recess - permanent engram archive in PostgreSQL."""

    level = StoreLevel.L3

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None

    @property
    async def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                _parse_db_url(self.settings.database_url),
                min_size=1,
                max_size=5,
                command_timeout=60,
            )
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await register_vector(conn)
                await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS l3_engrams (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                store_level TEXT DEFAULT 'L3',
                source TEXT DEFAULT 'user',
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                confidence REAL DEFAULT 0.5,
                version INT DEFAULT 1,
                checksum TEXT,
                lineage JSONB DEFAULT '[]',
                salience REAL DEFAULT 0.5,
                usage_count INT DEFAULT 0,
                authority REAL DEFAULT 0.5,
                metadata JSONB DEFAULT '{}',
                embedding vector(384),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_l3_ns ON l3_engrams(namespace)")
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_l3_embedding ON l3_engrams
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)
            """)
        except Exception:
            pass

    async def write(self, item: MemoryItem) -> MemoryItem:
        item.store_level = StoreLevel.L3
        pool = await self.pool
        emb_str = None
        if item.embedding:
            emb_str = "[" + ",".join(str(x) for x in item.embedding) + "]"

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO l3_engrams (
                    id, namespace, text, store_level, source, timestamp, confidence,
                    version, checksum, lineage, salience, usage_count, authority,
                    metadata, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::vector)
                ON CONFLICT (id) DO NOTHING
            """,
                item.id, item.namespace, item.text, item.store_level.value,
                item.source, item.timestamp, item.confidence, item.version,
                item.checksum, json.dumps(item.lineage), item.salience,
                item.usage_count, item.authority, json.dumps(item.metadata), emb_str,
            )
        return item

    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        pool = await self.pool
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        rows = await pool.fetch("""
            SELECT id, namespace, text, store_level, source, timestamp, confidence,
                   version, checksum, lineage, salience, usage_count, authority,
                   metadata, embedding
            FROM l3_engrams
            WHERE namespace = $1 AND confidence >= $2 AND embedding IS NOT NULL
            ORDER BY embedding <=> $3::vector
            LIMIT $4
        """, namespace, min_confidence, emb_str, limit)

        return [self._row_to_item(r) for r in rows]

    def _row_to_item(self, row) -> MemoryItem:
        emb = None
        if row.get("embedding"):
            emb = [float(x) for x in row["embedding"]]
        return MemoryItem(
            id=row["id"],
            namespace=row["namespace"],
            text=row["text"],
            store_level=StoreLevel.L3,
            source=row["source"],
            timestamp=row["timestamp"],
            confidence=float(row["confidence"]),
            version=row["version"],
            checksum=row["checksum"],
            lineage=row["lineage"] or [],
            salience=float(row["salience"]),
            usage_count=row["usage_count"],
            authority=float(row["authority"]),
            metadata=row["metadata"] or {},
            embedding=emb,
        )

    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        pool = await self.pool
        row = await pool.fetchrow(
            "SELECT * FROM l3_engrams WHERE id = $1 AND namespace = $2",
            item_id, namespace,
        )
        if row:
            return self._row_to_item(row)
        return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        pool = await self.pool
        result = await pool.execute(
            "DELETE FROM l3_engrams WHERE id = $1 AND namespace = $2",
            item_id, namespace,
        )
        return "DELETE 1" in result

    async def health(self) -> dict:
        try:
            pool = await self.pool
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy", "store": "L3"}
        except Exception as e:
            return {"status": "unhealthy", "store": "L3", "error": str(e)}

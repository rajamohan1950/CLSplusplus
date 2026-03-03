"""L2 Schema Graph - Neocortex equivalent.

Slow semantic integration, concept abstraction. Graph of concepts.
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


class L2SchemaGraph(BaseStore):
    """L2: Schema Graph - semantic concept graph with edge weights."""

    level = StoreLevel.L2

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._pool: Optional[asyncpg.Pool] = None

    @property
    async def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                _parse_db_url(self.settings.database_url),
                min_size=1,
                max_size=10,
                command_timeout=60,
            )
            async with self._pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await register_vector(conn)
                await self._init_schema(conn)
        return self._pool

    async def _init_schema(self, conn: asyncpg.Connection) -> None:
        """Create L2 graph tables."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS l2_nodes (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                text TEXT NOT NULL,
                store_level TEXT DEFAULT 'L2',
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
                metadata JSONB DEFAULT '{}',
                subject TEXT,
                predicate TEXT,
                object TEXT,
                embedding vector(384),
                edge_weight REAL DEFAULT 1.0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS l2_edges (
                id SERIAL PRIMARY KEY,
                namespace TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                predicate TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_l2_nodes_ns ON l2_nodes(namespace)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_l2_edges_ns ON l2_edges(namespace)")

    async def write(self, item: MemoryItem) -> MemoryItem:
        """Write node to schema graph."""
        item.store_level = StoreLevel.L2
        pool = await self.pool
        emb_str = None
        if item.embedding:
            emb_str = "[" + ",".join(str(x) for x in item.embedding) + "]"

        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO l2_nodes (
                    id, namespace, text, store_level, source, timestamp, confidence,
                    version, checksum, lineage, salience, usage_count, authority,
                    conflict_score, surprise, promotion_score, metadata,
                    subject, predicate, object, embedding, edge_weight
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21::vector, 1.0)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    confidence = EXCLUDED.confidence,
                    version = l2_nodes.version + 1,
                    salience = EXCLUDED.salience,
                    usage_count = l2_nodes.usage_count + 1,
                    promotion_score = EXCLUDED.promotion_score,
                    edge_weight = l2_nodes.edge_weight * 0.99 + 0.01
            """,
                item.id, item.namespace, item.text, item.store_level.value,
                item.source, item.timestamp, item.confidence, item.version,
                item.checksum, json.dumps(item.lineage), item.salience,
                item.usage_count, item.authority, item.conflict_score,
                item.surprise, item.promotion_score, json.dumps(item.metadata),
                item.subject, item.predicate, item.object, emb_str,
            )
        return item

    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        """Traverse graph + kNN on nodes."""
        pool = await self.pool
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        rows = await pool.fetch("""
            SELECT id, namespace, text, store_level, source, timestamp, confidence,
                   version, checksum, lineage, salience, usage_count, authority,
                   conflict_score, surprise, promotion_score, metadata,
                   subject, predicate, object, embedding
            FROM l2_nodes
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
            store_level=StoreLevel.L2,
            source=row["source"],
            timestamp=row["timestamp"],
            confidence=float(row["confidence"]),
            version=row["version"],
            checksum=row["checksum"],
            lineage=row["lineage"] or [],
            salience=float(row["salience"]),
            usage_count=row["usage_count"],
            authority=float(row["authority"]),
            conflict_score=float(row["conflict_score"]),
            surprise=float(row["surprise"]),
            promotion_score=float(row["promotion_score"]),
            metadata=row["metadata"] or {},
            subject=row.get("subject"),
            predicate=row.get("predicate"),
            object=row.get("object"),
            embedding=emb,
        )

    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        pool = await self.pool
        row = await pool.fetchrow(
            "SELECT * FROM l2_nodes WHERE id = $1 AND namespace = $2",
            item_id, namespace,
        )
        if row:
            return self._row_to_item(row)
        return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        pool = await self.pool
        await pool.execute("DELETE FROM l2_edges WHERE source_id = $1 OR target_id = $1", item_id, item_id)
        result = await pool.execute("DELETE FROM l2_nodes WHERE id = $1 AND namespace = $2", item_id, namespace)
        return "DELETE 1" in result

    async def list_for_sleep(self, namespace: str, limit: int = 20000) -> list[MemoryItem]:
        pool = await self.pool
        rows = await pool.fetch("""
            SELECT * FROM l2_nodes WHERE namespace = $1 ORDER BY timestamp DESC LIMIT $2
        """, namespace, limit)
        return [self._row_to_item(r) for r in rows]

    async def decay_edges(self, namespace: str, decay_factor: float = 0.95) -> int:
        """Apply edge weight decay for sleep cycle."""
        pool = await self.pool
        result = await pool.execute(
            "UPDATE l2_edges SET weight = weight * $1 WHERE namespace = $2",
            decay_factor, namespace,
        )
        return int(result.split()[-1]) if result else 0

    async def health(self) -> dict:
        try:
            pool = await self.pool
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return {"status": "healthy", "store": "L2"}
        except Exception as e:
            return {"status": "unhealthy", "store": "L2", "error": str(e)}

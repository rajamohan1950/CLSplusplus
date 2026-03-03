"""L3 Deep Recess - Thalamus/Basal Ganglia equivalent.

Stable long-term archive. MinIO + Parquet. Append-only, versioned.
"""

import io
import json
from datetime import datetime
from typing import Optional
from uuid import uuid4

from minio import Minio
import pyarrow as pa
import pyarrow.parquet as pq

from clsplusplus.config import Settings
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores.base import BaseStore


class L3DeepRecess(BaseStore):
    """L3: Deep Recess - permanent engram archive."""

    level = StoreLevel.L3

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self._client: Optional[Minio] = None
        self._memory_cache: dict[str, list[MemoryItem]] = {}  # namespace -> items (for kNN)

    @property
    def client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                secure=self.settings.minio_secure,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.settings.minio_bucket):
            self.client.make_bucket(self.settings.minio_bucket)

    def _object_key(self, namespace: str, item_id: str) -> str:
        return f"engrams/{namespace}/{item_id}.parquet"

    async def write(self, item: MemoryItem) -> MemoryItem:
        """Append engram to L3 - never overwrite, versioned."""
        item.store_level = StoreLevel.L3
        data = self._item_to_parquet(item)
        key = self._object_key(item.namespace, item.id)
        self.client.put_object(
            self.settings.minio_bucket,
            key,
            data,
            len(data.getvalue()),
            content_type="application/octet-stream",
        )
        # Update in-memory cache for reads (simplified - in prod use FAISS)
        if item.namespace not in self._memory_cache:
            self._memory_cache[item.namespace] = []
        self._memory_cache[item.namespace].append(item)
        return item

    def _item_to_parquet(self, item: MemoryItem) -> io.BytesIO:
        """Serialize item to Parquet bytes."""
        table = pa.table({
            "id": [item.id],
            "namespace": [item.namespace],
            "text": [item.text],
            "store_level": [item.store_level.value],
            "source": [item.source],
            "timestamp": [item.timestamp.isoformat()],
            "confidence": [item.confidence],
            "version": [item.version],
            "checksum": [item.checksum or ""],
            "lineage": [json.dumps(item.lineage)],
            "salience": [item.salience],
            "usage_count": [item.usage_count],
            "authority": [item.authority],
            "metadata": [json.dumps(item.metadata)],
            "embedding": [json.dumps(item.embedding or [])],
        })
        buf = io.BytesIO()
        pq.write_table(table, buf)
        buf.seek(0)
        return buf

    def _parquet_to_item(self, data: bytes) -> MemoryItem:
        """Deserialize Parquet to MemoryItem."""
        import json as j

        table = pq.read_table(io.BytesIO(data))
        ts = table.column("timestamp")[0]
        ts_str = str(ts) if ts else datetime.utcnow().isoformat()
        emb_col = table.column("embedding")[0]
        emb = j.loads(str(emb_col)) if emb_col and str(emb_col) != "[]" else None
        return MemoryItem(
            id=str(table.column("id")[0]),
            namespace=str(table.column("namespace")[0]),
            text=str(table.column("text")[0]),
            store_level=StoreLevel.L3,
            source=str(table.column("source")[0]),
            timestamp=datetime.fromisoformat(ts_str.replace("Z", "+00:00")),
            confidence=float(table.column("confidence")[0]),
            version=int(table.column("version")[0]),
            checksum=str(table.column("checksum")[0]) or None,
            lineage=j.loads(str(table.column("lineage")[0])),
            salience=float(table.column("salience")[0]),
            usage_count=int(table.column("usage_count")[0]),
            authority=float(table.column("authority")[0]),
            metadata=j.loads(str(table.column("metadata")[0])),
            embedding=emb,
        )

    async def read(
        self,
        query_embedding: list[float],
        namespace: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> list[MemoryItem]:
        """Read from L3 - semantic similarity (simplified: use cache + cosine)."""
        import numpy as np

        items = self._memory_cache.get(namespace, [])
        if not items:
            # Load from MinIO
            prefix = f"engrams/{namespace}/"
            objects = self.client.list_objects(self.settings.minio_bucket, prefix=prefix, recursive=True)
            for obj in objects:
                resp = self.client.get_object(self.settings.minio_bucket, obj.object_name)
                data = resp.read()
                resp.close()
                resp.release_connection()
                item = self._parquet_to_item(data)
                items.append(item)
            self._memory_cache[namespace] = items

        # Rank by cosine similarity
        q = np.array(query_embedding)
        scored = []
        for item in items:
            if item.embedding and item.confidence >= min_confidence:
                v = np.array(item.embedding)
                sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
                scored.append((sim, item))
        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:limit]]

    async def get_by_id(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        """Get engram by ID."""
        key = self._object_key(namespace, item_id)
        try:
            resp = self.client.get_object(self.settings.minio_bucket, key)
            data = resp.read()
            resp.close()
            resp.release_connection()
            return self._parquet_to_item(data)
        except Exception:
            return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        """Tombstone - in L3 we version, not delete. For RTBF, remove object."""
        key = self._object_key(namespace, item_id)
        try:
            self.client.remove_object(self.settings.minio_bucket, key)
            if namespace in self._memory_cache:
                self._memory_cache[namespace] = [i for i in self._memory_cache[namespace] if i.id != item_id]
            return True
        except Exception:
            return False

    async def health(self) -> dict:
        try:
            self.client.list_buckets()
            return {"status": "healthy", "store": "L3"}
        except Exception as e:
            return {"status": "unhealthy", "store": "L3", "error": str(e)}

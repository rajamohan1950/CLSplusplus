"""CLS++ Memory Service - orchestrates all stores and pipelines."""

import asyncio
import logging
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem, ReadRequest, ReadResponse, WriteRequest
from clsplusplus.plasticity import PlasticityEngine
from clsplusplus.reconsolidation import ReconsolidationGate
from clsplusplus.stores import L0WorkingBuffer, L1IndexingStore, L2SchemaGraph, L3PostgresStore

logger = logging.getLogger(__name__)


class MemoryService:
    """Main service - write flows to L0->L1, read from all stores."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)
        self.plasticity = PlasticityEngine(settings)
        self.reconsolidation = ReconsolidationGate(settings)
        self.l0 = L0WorkingBuffer(settings)
        self.l1 = L1IndexingStore(settings)
        self.l2 = L2SchemaGraph(settings)
        self.l3 = L3PostgresStore(settings)  # Postgres-backed (free tier, no MinIO)
        self._webhook_dispatcher = None  # Lazy init to avoid circular imports

    def _request_to_item(self, req: WriteRequest) -> MemoryItem:
        """Convert write request to MemoryItem."""
        return MemoryItem(
            text=req.text,
            namespace=req.namespace,
            source=req.source,
            salience=req.salience,
            authority=req.authority,
            metadata=req.metadata,
            subject=req.subject,
            predicate=req.predicate,
            object=req.object,
        )

    def _dispatch_webhook(self, event_type: str, item: MemoryItem) -> None:
        """Fire webhook event (fire-and-forget, never blocks)."""
        if self._webhook_dispatcher is None:
            return
        try:
            payload = {
                "id": item.id,
                "text": item.text,
                "namespace": item.namespace,
                "store_level": item.store_level.value if hasattr(item.store_level, 'value') else str(item.store_level),
                "confidence": item.confidence,
                "source": item.source,
            }
            asyncio.create_task(
                self._webhook_dispatcher.dispatch(event_type, payload, item.namespace)
            )
        except Exception:
            pass  # Webhook dispatch must never crash memory operations

    async def write(self, req: WriteRequest) -> MemoryItem:
        """Write memory: L0 (session buffer) + L1 (episodic persistence)."""
        item = self._request_to_item(req)
        item = self.embedding_service.embed_item(item)

        # L0: Working buffer for fast session access
        await self.l0.write(item)

        # L1: Episodic store - always persist for retrieval
        item.store_level = item.store_level  # Keep metadata
        await self.l1.write(item)

        # Webhook: fire memory.created event
        self._dispatch_webhook("memory.created", item)

        return item

    async def read(self, req: ReadRequest) -> ReadResponse:
        """Read from all stores, merge and rank by relevance."""
        query_emb = self.embedding_service.embed(req.query)
        all_items: list[MemoryItem] = []

        from clsplusplus.models import StoreLevel

        store_levels = req.store_levels or [StoreLevel.L0, StoreLevel.L1, StoreLevel.L2, StoreLevel.L3]

        if StoreLevel.L0 in store_levels:
            l0_items = await self.l0.read(
                query_emb, req.namespace, req.limit, req.min_confidence
            )
            all_items.extend(l0_items)

        if StoreLevel.L1 in store_levels:
            l1_items = await self.l1.read(
                query_emb, req.namespace, req.limit, req.min_confidence
            )
            all_items.extend(l1_items)

        if StoreLevel.L2 in store_levels:
            l2_items = await self.l2.read(
                query_emb, req.namespace, req.limit, req.min_confidence
            )
            all_items.extend(l2_items)

        if StoreLevel.L3 in store_levels:
            l3_items = await self.l3.read(
                query_emb, req.namespace, req.limit, req.min_confidence
            )
            all_items.extend(l3_items)

        # Dedupe by id, rank by confidence * store weight (L3 > L2 > L1 > L0)
        store_weight = {StoreLevel.L0: 0.5, StoreLevel.L1: 0.7, StoreLevel.L2: 0.9, StoreLevel.L3: 1.0}
        seen = set()
        unique = []
        for item in all_items:
            if item.id not in seen:
                seen.add(item.id)
                unique.append(item)

        unique.sort(
            key=lambda x: (x.confidence * store_weight.get(x.store_level, 0.5), -x.usage_count),
            reverse=True,
        )
        return ReadResponse(
            items=unique[: req.limit],
            query=req.query,
            namespace=req.namespace,
        )

    async def get_item(self, item_id: str, namespace: str) -> Optional[MemoryItem]:
        """Get single item by ID from any store."""
        for store in [self.l0, self.l1, self.l2, self.l3]:
            item = await store.get_by_id(item_id, namespace)
            if item:
                return item
        return None

    async def delete(self, item_id: str, namespace: str) -> bool:
        """Delete item from all stores. Returns True if deleted from at least one."""
        deleted = False
        for store in [self.l0, self.l1, self.l2, self.l3]:
            ok = await store.delete(item_id, namespace)
            deleted = deleted or ok

        # Webhook: fire memory.deleted event
        if deleted and self._webhook_dispatcher:
            try:
                payload = {"id": item_id, "namespace": namespace}
                asyncio.create_task(
                    self._webhook_dispatcher.dispatch("memory.deleted", payload, namespace)
                )
            except Exception:
                pass

        return deleted

    async def health(self) -> dict:
        """Aggregate health of all stores."""
        l0_h = await self.l0.health()
        l1_h = await self.l1.health()
        l2_h = await self.l2.health()
        l3_h = await self.l3.health()
        return {
            "status": "healthy" if all(s["status"] == "healthy" for s in [l0_h, l1_h, l2_h, l3_h]) else "degraded",
            "stores": {"L0": l0_h, "L1": l1_h, "L2": l2_h, "L3": l3_h},
        }

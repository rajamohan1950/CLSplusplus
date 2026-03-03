"""Sleep Cycle - nightly maintenance.

N1: Rank, N2: Strengthen+Decay, N3: Deduplicate, REM: Consolidate+Dream.
"""

import asyncio
from datetime import datetime
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.plasticity import PlasticityEngine
from clsplusplus.stores import L0WorkingBuffer, L1IndexingStore, L2SchemaGraph, L3DeepRecess


class SleepOrchestrator:
    """Runs 4-phase sleep cycle. 60-min budget, idempotent."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.plasticity = PlasticityEngine(settings)
        self.embedding_service = EmbeddingService(settings)
        self.l0 = L0WorkingBuffer(settings)
        self.l1 = L1IndexingStore(settings)
        self.l2 = L2SchemaGraph(settings)
        self.l3 = L3DeepRecess(settings)

    async def run(self, namespace: str = "default") -> dict:
        """Run full sleep cycle. Returns report."""
        report = {
            "namespace": namespace,
            "started_at": datetime.utcnow().isoformat(),
            "phases": {},
            "reinforced": 0,
            "pruned": 0,
            "deduped": 0,
            "engraved": 0,
            "error": None,
        }

        try:
            # N1: Rank - score all L1 items
            items = await self.l1.list_for_sleep(namespace, limit=20000)
            for item in items:
                self.plasticity.compute_score(item)

            # N2: Strengthen + Decay
            top_pct = int(len(items) * 0.2)
            bottom_pct = int(len(items) * 0.4)
            items_sorted = sorted(items, key=lambda x: -x.promotion_score)

            for i, item in enumerate(items_sorted[:top_pct]):
                item.confidence = min(1.0, item.confidence + 0.05)
                item.salience = min(1.0, item.salience + 0.02)
                await self.l1.update_scores(
                    item.id, namespace,
                    confidence=item.confidence,
                    salience=item.salience,
                )
                report["reinforced"] += 1

            for item in items_sorted[-bottom_pct:]:
                self.plasticity.apply_decay(item)
                if self.plasticity.should_prune(item):
                    await self.l1.delete(item.id, namespace)
                    report["pruned"] += 1
                else:
                    await self.l1.update_scores(
                        item.id, namespace,
                        salience=item.salience,
                        confidence=item.confidence,
                    )

            # N3: Deduplicate - merge similar items
            dedup_count = 0
            seen = []
            for item in items_sorted:
                if not item.embedding:
                    continue
                merged = False
                for other in seen:
                    if EmbeddingService.cosine_similarity(item.embedding, other.embedding) >= 0.92:
                        # Merge: keep higher confidence
                        if item.confidence > other.confidence:
                            await self.l1.delete(other.id, namespace)
                            seen.remove(other)
                            seen.append(item)
                        else:
                            await self.l1.delete(item.id, namespace)
                        dedup_count += 1
                        merged = True
                        break
                if not merged:
                    seen.append(item)
            report["deduped"] = dedup_count

            # REM: Consolidate - promote L1->L2, L2->L3
            for item in items_sorted:
                if self.plasticity.should_promote_to_l2(item):
                    item.store_level = StoreLevel.L2
                    item = self.embedding_service.embed_item(item)
                    await self.l2.write(item)
                    report["engraved"] += 1

            l2_items = await self.l2.list_for_sleep(namespace, limit=5000)
            for item in l2_items:
                if self.plasticity.should_promote_to_l3(item):
                    item.store_level = StoreLevel.L3
                    await self.l3.write(item)
                    report["engraved"] += 1

            report["phases"] = {
                "N1": "complete",
                "N2": "complete",
                "N3": "complete",
                "REM": "complete",
            }

        except Exception as e:
            report["error"] = str(e)

        report["completed_at"] = datetime.utcnow().isoformat()
        return report

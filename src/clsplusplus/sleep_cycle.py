"""Sleep Cycle - hippocampal replay and schema consolidation.

REM phase only:
  1. recall_long_tail() — hippocampal replay: keep low-retrieval items alive
  2. Schema export — write crystallized solid/glass items from PhaseMemoryEngine to L2
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from clsplusplus.config import Settings
from clsplusplus.embeddings import EmbeddingService
from clsplusplus.memory_phase import PhaseMemoryEngine
from clsplusplus.models import MemoryItem, StoreLevel
from clsplusplus.stores import L2SchemaGraph

logger = logging.getLogger(__name__)


class SleepOrchestrator:
    """Runs REM phase: hippocampal replay + schema persistence."""

    def __init__(self, settings: Optional[Settings] = None, engine: Optional[PhaseMemoryEngine] = None):
        self.settings = settings or Settings()
        self.embedding_service = EmbeddingService(settings)
        self.l2 = L2SchemaGraph(settings)
        # Use shared engine if provided (from MemoryService), else create own
        self.engine = engine or PhaseMemoryEngine()

    async def run(self, namespace: str = "default") -> dict:
        """Run REM sleep cycle. Returns report."""
        report = {
            "namespace": namespace,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phase": "REM",
            "rehearsed": 0,
            "schemas_exported": 0,
            "error": None,
        }

        try:
            # REM-1: Hippocampal replay — rehearse long-tail items to keep them alive
            rehearsed = self.engine.recall_long_tail(namespace, batch_size=50)
            report["rehearsed"] = rehearsed

            # REM-2: Schema export — write crystallized schemas to L2 for persistence
            items = self.engine._items.get(namespace, [])
            for phase_item in items:
                if phase_item.schema_meta is None:
                    continue
                if phase_item.consolidation_strength < self.engine.STRENGTH_FLOOR:
                    continue
                try:
                    mem_item = MemoryItem(
                        id=phase_item.id,
                        text=phase_item.fact.raw_text,
                        namespace=phase_item.namespace,
                        store_level=StoreLevel.L2,
                        confidence=min(1.0, phase_item.consolidation_strength),
                        salience=phase_item.surprise_at_birth,
                        usage_count=phase_item.retrieval_count,
                        surprise=phase_item.surprise_at_birth,
                        subject=phase_item.fact.subject or None,
                        predicate=phase_item.fact.relation or None,
                        object=(phase_item.fact.value[:256] if phase_item.fact.value else None),
                    )
                    mem_item = self.embedding_service.embed_item(mem_item)
                    await self.l2.write(mem_item)
                    report["schemas_exported"] += 1
                except Exception as e:
                    logger.warning("Schema export to L2 failed for %s: %s", phase_item.id, e)

        except Exception as e:
            logger.error("Sleep cycle error for namespace '%s': %s", namespace, e)
            report["error"] = str(e)

        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        return report

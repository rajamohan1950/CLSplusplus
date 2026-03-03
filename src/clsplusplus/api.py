"""CLS++ REST API - FastAPI application."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.models import (
    AdjudicateRequest,
    HealthResponse,
    ReadRequest,
    ReadResponse,
    WriteRequest,
)
from clsplusplus.sleep_cycle import SleepOrchestrator


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create FastAPI application."""
    settings = settings or Settings()
    memory_service = MemoryService(settings)
    sleep_orchestrator = SleepOrchestrator(settings)

    app = FastAPI(
        title="CLS++ API",
        description="Brain-inspired, model-agnostic persistent memory for LLMs",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/memory/write")
    async def write_memory(req: WriteRequest):
        """Write memory. Flows to L0, promotes to L1 if score warrants."""
        item = await memory_service.write(req)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text}

    @app.post("/v1/memory/read", response_model=ReadResponse)
    async def read_memory(req: ReadRequest):
        """Read memories by semantic query across all stores."""
        return await memory_service.read(req)

    @app.get("/v1/memory/item/{item_id}")
    async def get_item(item_id: str, namespace: str = "default"):
        """Get full item with lineage and versions."""
        item = await memory_service.get_item(item_id, namespace)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item.to_dict()

    @app.post("/v1/memory/sleep")
    async def trigger_sleep(namespace: str = "default"):
        """Trigger nightly sleep cycle (admin)."""
        report = await sleep_orchestrator.run(namespace)
        return report

    @app.post("/v1/memory/adjudicate_conflict")
    async def adjudicate_conflict(req: AdjudicateRequest):
        """Submit conflicting fact + evidence for reconsolidation gate."""
        from clsplusplus.models import MemoryItem

        new_item = MemoryItem(text=req.new_fact, namespace=req.namespace)
        new_item = memory_service.embedding_service.embed_item(new_item)
        old_item = None
        if req.existing_item_id:
            old_item = await memory_service.get_item(req.existing_item_id, req.namespace)
        if old_item and memory_service.reconsolidation.should_overwrite(
            new_item, old_item, req.evidence
        ):
            await memory_service.l1.write(new_item)
            return {"decision": "overwrite", "new_id": new_item.id}
        if not old_item:
            await memory_service.l1.write(new_item)
            return {"decision": "accepted", "new_id": new_item.id}
        return {"decision": "reject", "reason": "Insufficient evidence quorum"}

    @app.get("/v1/memory/health", response_model=HealthResponse)
    async def health():
        """Composite health + per-store metrics."""
        h = await memory_service.health()
        return HealthResponse(
            status=h["status"],
            stores=h["stores"],
        )

    return app


app = create_app()

"""CLS++ REST API - FastAPI application."""

from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService
from clsplusplus.middleware import AuthMiddleware, RateLimitMiddleware
from clsplusplus.models import (
    AdjudicateRequest,
    DemoChatRequest,
    ForgetRequest,
    HealthResponse,
    ReadRequest,
    ReadResponse,
    WriteRequest,
)
from clsplusplus.models import _validate_item_id as validate_item_id
from clsplusplus.models import _validate_namespace as validate_namespace
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
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(AuthMiddleware, settings=settings)

    def _ns_query(default: str = "default") -> str:
        return Query(default=default, min_length=1, max_length=64)

    @app.get("/")
    async def root():
        """API root - links to docs and health."""
        return {
            "name": "CLS++ API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/v1/memory/health",
        }

    @app.get("/health")
    async def health_redirect():
        """Redirect /health to /v1/memory/health for convenience."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/v1/memory/health")

    @app.post("/v1/memory/write")
    async def write_memory(req: WriteRequest):
        """Write memory. Flows to L0, promotes to L1 if score warrants."""
        item = await memory_service.write(req)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text}

    @app.post("/v1/memories/encode")
    async def encode_memory(req: WriteRequest):
        """Product alias: POST /memories/encode -> write."""
        item = await memory_service.write(req)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text}

    @app.post("/v1/memory/read", response_model=ReadResponse)
    async def read_memory(req: ReadRequest):
        """Read memories by semantic query across all stores."""
        return await memory_service.read(req)

    @app.post("/v1/memories/retrieve", response_model=ReadResponse)
    async def retrieve_memories(req: ReadRequest):
        """Product alias: POST /memories/retrieve -> read."""
        return await memory_service.read(req)

    @app.get("/v1/memory/item/{item_id}")
    async def get_item(
        item_id: str = Path(..., min_length=1, max_length=64),
        namespace: str = _ns_query(),
    ):
        """Get full item with lineage and versions."""
        try:
            validate_item_id(item_id)
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        item = await memory_service.get_item(item_id, namespace)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item.to_dict()

    @app.post("/v1/memory/sleep")
    async def trigger_sleep(namespace: str = _ns_query()):
        """Trigger nightly sleep cycle (admin)."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        report = await sleep_orchestrator.run(namespace)
        return report

    @app.post("/v1/memories/consolidate")
    async def consolidate_memories(namespace: str = _ns_query()):
        """Product alias: POST /memories/consolidate -> sleep."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        report = await sleep_orchestrator.run(namespace)
        return report

    @app.delete("/v1/memory/forget")
    async def forget_memory(req: ForgetRequest):
        """Delete a memory by ID (RTBF)."""
        deleted = await memory_service.delete(req.item_id, req.namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"deleted": True, "item_id": req.item_id}

    @app.delete("/v1/memories/forget")
    async def forget_memory_alias(req: ForgetRequest):
        """Product alias: DELETE /memories/forget."""
        deleted = await memory_service.delete(req.item_id, req.namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"deleted": True, "item_id": req.item_id}

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

    @app.get("/v1/demo/status")
    async def demo_status():
        """Check which LLM keys are configured (for debugging)."""
        return {
            "claude": bool(getattr(settings, "anthropic_api_key", None)),
            "openai": bool(getattr(settings, "openai_api_key", None)),
            "gemini": bool(getattr(settings, "google_api_key", None)),
        }

    @app.post("/v1/demo/chat")
    async def demo_chat(req: DemoChatRequest):
        """
        Real LLM demo: Claude, OpenAI, or Gemini with shared CLS++ memory.
        Requires CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY, CLS_GOOGLE_API_KEY in env.
        """
        from clsplusplus.demo_llm import chat_with_llm

        if req.model not in ("claude", "openai", "gemini"):
            raise HTTPException(status_code=400, detail="model must be claude, openai, or gemini")

        try:
            reply = await chat_with_llm(
                memory_service, settings, req.model, req.message.strip(), req.namespace
            )
            return {"model": req.model, "reply": reply}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Demo error: {str(e)[:200]}",
            )

    @app.get("/v1/memory/health", response_model=HealthResponse)
    async def health():
        """Composite health + per-store metrics."""
        h = await memory_service.health()
        stores = dict(h["stores"])
        if h["status"] == "degraded" and "localhost" in str(stores):
            stores["_hint"] = {
                "status": "info",
                "store": "Setup",
                "error": "Add CLS_REDIS_URL and CLS_DATABASE_URL in Render Dashboard.",
            }
        return HealthResponse(status=h["status"], stores=stores)

    @app.get("/v1/health/score", response_model=HealthResponse)
    async def health_score():
        """Product alias: GET /health/score -> memory health."""
        return await health()

    @app.get("/v1/memories/knowledge", response_model=ReadResponse)
    async def query_knowledge(
        query: str = Query(..., min_length=1, max_length=4096),
        namespace: str = _ns_query(),
        limit: int = Query(default=10, ge=1, le=100),
    ):
        """Product alias: GET /knowledge - query L2/L3 (neocortical) only."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        from clsplusplus.models import ReadRequest, StoreLevel

        req = ReadRequest(
            query=query,
            namespace=namespace,
            limit=limit,
            store_levels=[StoreLevel.L2, StoreLevel.L3],
        )
        return await memory_service.read(req)

    return app


app = create_app()

"""CLS++ REST API - FastAPI application."""

from typing import Optional

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from clsplusplus.config import Settings
from clsplusplus.integration_service import IntegrationService
from clsplusplus.memory_service import MemoryService
from clsplusplus.middleware import AuthMiddleware, RateLimitMiddleware, RequestIdMiddleware
from clsplusplus.models import (
    AdjudicateRequest,
    ApiKeyCreate,
    DemoChatRequest,
    ForgetRequest,
    HealthResponse,
    IntegrationCreate,
    MemoryCycleRequest,
    ReadRequest,
    ReadResponse,
    WebhookCreate,
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
    integration_service = IntegrationService(settings)

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
    # Middleware execution order: outermost (added last) runs first.
    # RequestId first -> RateLimit -> Auth -> route handler
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(RequestIdMiddleware)

    # Structured error handler (blueprint: error messages that teach)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, str):
            content = {
                "error": "request_error",
                "message": detail,
                "status_code": exc.status_code,
            }
            if exc.status_code == 401:
                content["fix"] = "Add Authorization: Bearer <api_key> header"
                content["docs"] = "https://github.com/rajamohan1950/CLSplusplus/wiki/API-Reference"
            elif exc.status_code == 429:
                content["fix"] = f"Retry after {request.headers.get('Retry-After', 60)} seconds or upgrade plan"
                content["docs"] = "https://github.com/rajamohan1950/CLSplusplus/wiki/SaaS-and-Pricing"
            elif exc.status_code == 422:
                content["fix"] = "Check request body against API schema"
                content["docs"] = "/docs"
        else:
            content = {"error": "request_error", "message": str(detail), "status_code": exc.status_code}
        return JSONResponse(status_code=exc.status_code, content=content)

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

    async def _record_usage(operation: str, request: Request):
        """Fire-and-forget usage tracking. Must never crash a user request."""
        try:
            api_key = getattr(request.state, "api_key", None)
            if api_key and settings.track_usage:
                from clsplusplus.usage import record_usage
                await record_usage(api_key, operation, settings)
        except Exception:
            pass  # Usage tracking failure must not affect user responses

    @app.post("/v1/memory/write")
    async def write_memory(req: WriteRequest, request: Request):
        """Write memory. Flows to L0, promotes to L1 if score warrants."""
        item = await memory_service.write(req)
        await _record_usage("write", request)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text}

    @app.post("/v1/memories/encode")
    async def encode_memory(req: WriteRequest, request: Request):
        """Product alias: POST /memories/encode -> write."""
        item = await memory_service.write(req)
        await _record_usage("encode", request)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text}

    @app.post("/v1/memory/read", response_model=ReadResponse)
    async def read_memory(req: ReadRequest, request: Request):
        """Read memories by semantic query across all stores."""
        result = await memory_service.read(req)
        await _record_usage("read", request)
        return result

    @app.post("/v1/memories/retrieve", response_model=ReadResponse)
    async def retrieve_memories(req: ReadRequest, request: Request):
        """Product alias: POST /memories/retrieve -> read."""
        result = await memory_service.read(req)
        await _record_usage("retrieve", request)
        return result

    @app.post("/v1/memories/search", response_model=ReadResponse)
    async def search_memories(req: ReadRequest, request: Request):
        """Resource-oriented: POST /memories/search -> read."""
        result = await memory_service.read(req)
        await _record_usage("retrieve", request)
        return result

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
            import logging
            logging.getLogger(__name__).error("Demo chat error: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Demo error: An internal error occurred. Check server logs.",
            )

    @app.post("/v1/demo/memory-cycle")
    async def memory_cycle(req: MemoryCycleRequest):
        """Run full memory lifecycle: encode → retrieve → augment → cross-session.

        Proves memory persists across models and sessions.
        """
        for m in req.models:
            if m not in ("claude", "openai", "gemini"):
                raise HTTPException(status_code=400, detail=f"Invalid model: {m}. Use claude, openai, or gemini.")

        from clsplusplus.memory_cycle import run_memory_cycle

        try:
            result = await run_memory_cycle(
                memory_service, settings,
                statements=req.statements,
                queries=req.queries,
                models=req.models,
                namespace=req.namespace,
            )
            return result
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Memory cycle error: %s", e)
            raise HTTPException(
                status_code=500,
                detail="Memory cycle error: An internal error occurred. Check server logs.",
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
        request: Request,
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
        result = await memory_service.read(req)
        await _record_usage("knowledge", request)
        return result

    @app.delete("/v1/memories/{item_id}")
    async def forget_memory_by_id(
        item_id: str = Path(..., min_length=1, max_length=64),
        namespace: str = _ns_query(),
    ):
        """Resource-oriented: DELETE /memories/{id} (forget)."""
        try:
            validate_item_id(item_id)
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        deleted = await memory_service.delete(item_id, namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        return {"deleted": True, "item_id": item_id}

    @app.get("/v1/usage")
    async def usage_endpoint(request: Request):
        """Usage metrics for current period (marketplace billing). Requires API key when auth enabled."""
        from clsplusplus.auth import get_api_key_from_request
        api_key = getattr(request.state, "api_key", None) or get_api_key_from_request(request.headers.get("Authorization"))
        if settings.require_api_key and not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        from clsplusplus.usage import get_usage as _get_usage
        return await _get_usage(api_key or "anonymous", settings)

    @app.get("/v1/billing/usage")
    async def billing_usage(request: Request):
        """Billing API: usage for current period (alias for /v1/usage)."""
        return await usage_endpoint(request)

    # =========================================================================
    # Integration Management API — Self-service integration endpoints
    # =========================================================================

    @app.post("/v1/integrations")
    async def create_integration(req: IntegrationCreate):
        """Register a new integration. Returns integration + first API key."""
        integration, api_key = await integration_service.register(req)
        return {
            "integration": integration.model_dump(mode="json"),
            "api_key": api_key.model_dump(mode="json"),
            "_hint": "Save your API key now — it won't be shown again.",
        }

    @app.get("/v1/integrations")
    async def list_integrations(namespace: str = _ns_query()):
        """List all integrations for a namespace."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        items = await integration_service.list_all(namespace)
        return {"integrations": [i.model_dump(mode="json") for i in items]}

    @app.get("/v1/integrations/{integration_id}")
    async def get_integration(
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Get integration details."""
        result = await integration_service.get(integration_id)
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        return result.model_dump(mode="json")

    @app.delete("/v1/integrations/{integration_id}")
    async def delete_integration(
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Deactivate an integration (revokes all keys, disables webhooks)."""
        deleted = await integration_service.delete(integration_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Integration not found")
        return {"deleted": True, "integration_id": integration_id}

    # --- API Keys ---

    @app.post("/v1/integrations/{integration_id}/keys")
    async def create_api_key(
        req: ApiKeyCreate,
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Create a new scoped API key. Key is shown only once."""
        try:
            result = await integration_service.create_key(integration_id, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        return {
            "api_key": result.model_dump(mode="json"),
            "_hint": "Save your API key now — it won't be shown again.",
        }

    @app.get("/v1/integrations/{integration_id}/keys")
    async def list_api_keys(
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """List API keys for an integration (keys are masked)."""
        keys = await integration_service.list_keys(integration_id)
        return {"keys": [k.model_dump(mode="json") for k in keys]}

    @app.post("/v1/integrations/{integration_id}/keys/{key_id}/rotate")
    async def rotate_api_key(
        integration_id: str = Path(..., min_length=1, max_length=64),
        key_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Rotate an API key. Old key has 24h grace period."""
        result = await integration_service.rotate_key(key_id)
        if not result:
            raise HTTPException(status_code=404, detail="API key not found or already revoked")
        return {
            "new_key": result.model_dump(mode="json"),
            "_hint": "Old key is valid for 24 more hours. Save the new key now.",
        }

    @app.delete("/v1/integrations/{integration_id}/keys/{key_id}")
    async def revoke_api_key(
        integration_id: str = Path(..., min_length=1, max_length=64),
        key_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Revoke an API key immediately."""
        revoked = await integration_service.revoke_key(key_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="API key not found or already revoked")
        return {"revoked": True, "key_id": key_id}

    # --- Webhooks ---

    @app.post("/v1/integrations/{integration_id}/webhooks")
    async def create_webhook(
        req: WebhookCreate,
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Subscribe to webhook events. Signing secret shown only once."""
        try:
            result = await integration_service.subscribe_webhook(integration_id, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not result:
            raise HTTPException(status_code=404, detail="Integration not found")
        return {
            "webhook": result.model_dump(mode="json"),
            "_hint": "Save your webhook signing secret now — it won't be shown again.",
        }

    @app.get("/v1/integrations/{integration_id}/webhooks")
    async def list_webhooks(
        integration_id: str = Path(..., min_length=1, max_length=64),
    ):
        """List webhook subscriptions for an integration."""
        webhooks = await integration_service.list_webhooks(integration_id)
        return {"webhooks": [w.model_dump(mode="json") for w in webhooks]}

    @app.delete("/v1/integrations/{integration_id}/webhooks/{webhook_id}")
    async def delete_webhook(
        integration_id: str = Path(..., min_length=1, max_length=64),
        webhook_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Unsubscribe from webhook events."""
        deleted = await integration_service.unsubscribe_webhook(webhook_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"deleted": True, "webhook_id": webhook_id}

    # --- Audit Events ---

    @app.get("/v1/integrations/{integration_id}/events")
    async def list_integration_events(
        integration_id: str = Path(..., min_length=1, max_length=64),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """Get audit log for an integration."""
        events = await integration_service.get_events(integration_id, limit)
        return {"events": [e.model_dump(mode="json") for e in events]}

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanly close all connection pools on shutdown."""
        for store in [memory_service.l0, memory_service.l1, memory_service.l2, memory_service.l3]:
            if hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:
                    pass
        try:
            await integration_service.close()
        except Exception:
            pass

    return app


app = create_app()

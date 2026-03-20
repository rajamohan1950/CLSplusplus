"""CLS++ REST API - FastAPI application."""

import asyncio
import logging
import time as _time
import uuid as _uuid_mod
from dataclasses import dataclass, field
from typing import Optional, List

import os
from pathlib import Path as FilePath

from fastapi import FastAPI, HTTPException, Path, Query, Request, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from clsplusplus.config import Settings
from clsplusplus.integration_service import IntegrationService
from clsplusplus.memory_service import MemoryService
from clsplusplus.middleware import AuthMiddleware, RateLimitMiddleware, RequestIdMiddleware, TracingMiddleware
from clsplusplus.tracer import tracer
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
    sleep_orchestrator = SleepOrchestrator(settings, engine=memory_service.engine)
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
    # TracingMiddleware → RequestId → RateLimit → Auth → route handler
    app.add_middleware(AuthMiddleware, settings=settings)
    app.add_middleware(RateLimitMiddleware, settings=settings)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(TracingMiddleware)  # outermost: traces every /v1/* request

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

    def _trace_id(request: Request) -> str:
        """Return the trace ID already set by TracingMiddleware, or generate one."""
        return (
            getattr(request.state, "trace_id", None)
            or request.headers.get("x-trace-id")
            or request.headers.get("x-request-id")
            or getattr(request.state, "request_id", None)
            or str(__import__("uuid").uuid4())
        )

    # Detect website directory (in Docker: /app/website, local dev: ../website relative to src)
    _website_dir = os.environ.get("CLS_WEBSITE_DIR")
    if not _website_dir:
        _candidate = FilePath(__file__).resolve().parent.parent.parent / "website"
        if _candidate.is_dir():
            _website_dir = str(_candidate)

    @app.get("/")
    async def root():
        """Serve index.html if website is bundled, otherwise API info JSON."""
        if _website_dir:
            index = FilePath(_website_dir) / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
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
        tid = _trace_id(request)
        with tracer.span(tid, "api.write", "api",
                         input=req.text[:200],
                         namespace=req.namespace, source=req.source) as api_hop:
            item = await memory_service.write(req, trace_id=tid)
            tracer.add_metadata(tid, api_hop,
                                output=f"item_id={str(item.id)[:8]}…  level={item.store_level.value}")
        await _record_usage("write", request)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text, "trace_id": tid}

    @app.post("/v1/memories/encode")
    async def encode_memory(req: WriteRequest, request: Request):
        """Product alias: POST /memories/encode -> write."""
        tid = _trace_id(request)
        with tracer.span(tid, "api.encode", "api",
                         input=req.text[:200],
                         namespace=req.namespace) as api_hop:
            item = await memory_service.write(req, trace_id=tid)
            tracer.add_metadata(tid, api_hop,
                                output=f"item_id={str(item.id)[:8]}…  level={item.store_level.value}")
        await _record_usage("encode", request)
        return {"id": item.id, "store_level": item.store_level.value, "text": item.text, "trace_id": tid}

    @app.post("/v1/memory/read", response_model=ReadResponse)
    async def read_memory(req: ReadRequest, request: Request):
        """Read memories by semantic query across all stores."""
        tid = _trace_id(request)
        with tracer.span(tid, "api.read", "api",
                         input=req.query[:200],
                         namespace=req.namespace, limit=req.limit) as api_hop:
            result = await memory_service.read(req, trace_id=tid)
            items = result.items or []
            preview = items[0].text[:80] if items else "no results"
            tracer.add_metadata(tid, api_hop, output=f"{len(items)} items: {preview}")
        await _record_usage("read", request)
        result.trace_id = tid
        return result

    @app.post("/v1/memories/retrieve", response_model=ReadResponse)
    async def retrieve_memories(req: ReadRequest, request: Request):
        """Product alias: POST /memories/retrieve -> read."""
        tid = _trace_id(request)
        with tracer.span(tid, "api.retrieve", "api",
                         input=req.query[:200],
                         namespace=req.namespace) as api_hop:
            result = await memory_service.read(req, trace_id=tid)
            items = result.items or []
            tracer.add_metadata(tid, api_hop, output=f"{len(items)} items")
        await _record_usage("retrieve", request)
        result.trace_id = tid
        return result

    @app.post("/v1/memories/search", response_model=ReadResponse)
    async def search_memories(req: ReadRequest, request: Request):
        """Resource-oriented: POST /memories/search -> read."""
        tid = _trace_id(request)
        with tracer.span(tid, "api.search", "api",
                         input=req.query[:200],
                         namespace=req.namespace) as api_hop:
            result = await memory_service.read(req, trace_id=tid)
            items = result.items or []
            tracer.add_metadata(tid, api_hop, output=f"{len(items)} items")
        await _record_usage("retrieve", request)
        result.trace_id = tid
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

    @app.post("/v1/memories/prewarm")
    async def prewarm_namespace(namespace: str = _ns_query()):
        """Pre-load a namespace into memory so the first user request is instant.

        Call this at application startup for active namespaces.  Returns immediately
        — loading happens in the background.  Idempotent (safe to call repeatedly).
        """
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        await memory_service.prewarm(namespace)
        already = namespace in memory_service._loaded_namespaces
        return {"status": "loaded" if already else "loading", "namespace": namespace}

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
        result = await memory_service.adjudicate(
            new_text=req.new_fact,
            namespace=req.namespace,
            evidence=req.evidence,
            existing_item_id=req.existing_item_id,
        )
        # If the returned item is the same as the existing one, quorum was not met
        decision = "rejected" if (req.existing_item_id and result.id == req.existing_item_id) else "accepted"
        return {"decision": decision, "new_id": result.id}

    @app.get("/v1/demo/status")
    async def demo_status():
        """Check which LLM keys are configured (for debugging)."""
        return {
            "claude": bool(getattr(settings, "anthropic_api_key", None)),
            "openai": bool(getattr(settings, "openai_api_key", None)),
            "gemini": bool(getattr(settings, "google_api_key", None)),
        }

    @app.post("/v1/demo/chat")
    async def demo_chat(req: DemoChatRequest, request: Request):
        """
        Real LLM demo: Claude, OpenAI, or Gemini with shared CLS++ memory.
        Requires CLS_ANTHROPIC_API_KEY, CLS_OPENAI_API_KEY, CLS_GOOGLE_API_KEY in env.
        """
        from clsplusplus.demo_llm import chat_with_llm

        if req.model not in ("claude", "openai", "gemini"):
            raise HTTPException(status_code=400, detail="model must be claude, openai, or gemini")

        tid = _trace_id(request)
        try:
            with tracer.span(tid, "api.demo_chat", "api",
                             input=req.message[:200],
                             model=req.model, namespace=req.namespace) as api_hop:
                reply = await chat_with_llm(
                    memory_service, settings, req.model, req.message.strip(), req.namespace,
                    trace_id=tid,
                )
                tracer.add_metadata(tid, api_hop, output=reply[:200])
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

    # =========================================================================
    # Chat Session API — /v1/chat/sessions
    # =========================================================================

    @dataclass
    class _ChatMsg:
        role: str          # "user" | "assistant"
        content: str
        memory_used: bool = False
        memory_count: int = 0

    @dataclass
    class _ChatSession:
        session_id: str
        name: str
        namespace: str
        messages: list = field(default_factory=list)
        created_at: float = field(default_factory=_time.time)

    # In-memory session store — persists until server restart
    _sessions: dict = {}
    _session_counter: list = [0]  # mutable counter via list trick

    def _pick_model() -> str:
        """Return first available LLM model based on configured API keys."""
        if getattr(settings, "anthropic_api_key", None):
            return "claude"
        if getattr(settings, "openai_api_key", None):
            return "openai"
        if getattr(settings, "google_api_key", None):
            return "gemini"
        return "claude"

    def _phase_dynamics(namespace: str) -> dict:
        """Build the thermodynamic debug snapshot for a namespace."""
        raw_items = memory_service.engine._items.get(namespace, [])
        floor = memory_service.engine.STRENGTH_FLOOR
        tau_c1 = memory_service.engine.TAU_C1
        event_counter = memory_service.engine._event_counter

        total_F = 0.0
        liquid_count = 0
        gas_count = 0
        item_list = []

        for item in raw_items:
            d = item.to_debug_dict(strength_floor=floor)
            total_F += d["free_energy"]
            if d["phase"] in ("liquid", "solid", "glass"):
                liquid_count += 1
            else:
                gas_count += 1
            item_list.append(d)

        # Most relevant first (highest consolidation_strength)
        item_list.sort(key=lambda x: x["consolidation_strength"], reverse=True)
        n = len(raw_items)
        rho = n / max(n + tau_c1, 1e-9)

        return {
            "memory_density_rho": round(rho, 6),
            "global_event_counter": event_counter,
            "total_free_energy": round(total_F, 4),
            "tau_c1": tau_c1,
            "liquid_count": liquid_count,
            "gas_count": gas_count,
            "items": item_list[:20],  # top 20 by strength
        }

    class _ChatMsgReq(BaseModel):
        message: str

    @app.post("/v1/chat/sessions")
    async def create_chat_session():
        """Create a new chat session. Returns session_id + auto-generated name."""
        sid = str(_uuid_mod.uuid4())
        _session_counter[0] += 1
        name = f"Chat {_session_counter[0]}"
        ns = f"chat-{sid[:12]}"
        _sessions[sid] = _ChatSession(session_id=sid, name=name, namespace=ns)
        return {"session_id": sid, "name": name}

    @app.get("/v1/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str = Path(..., min_length=1, max_length=64)):
        """Return a session with its full message history."""
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {
            "session_id": sess.session_id,
            "name": sess.name,
            "messages": [
                {"role": m.role, "content": m.content,
                 "memory_used": m.memory_used, "memory_count": m.memory_count}
                for m in sess.messages
            ],
        }

    @app.delete("/v1/chat/sessions/{session_id}")
    async def delete_chat_session(session_id: str = Path(..., min_length=1, max_length=64)):
        """Delete a session (removes history but not memory)."""
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        del _sessions[session_id]
        return {"deleted": True, "session_id": session_id}

    @app.post("/v1/chat/sessions/{session_id}/message")
    async def chat_session_message(
        req: _ChatMsgReq,
        request: Request,
        session_id: str = Path(..., min_length=1, max_length=64),
    ):
        """Send a user message. Returns LLM reply + full debug snapshot."""
        from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini

        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        tid = _trace_id(request)
        user_text = req.message.strip()
        if not user_text:
            raise HTTPException(status_code=400, detail="message must not be empty")

        ns = sess.namespace
        model = _pick_model()

        # 1. Store user message to memory (skip pure questions)
        def _is_question(t: str) -> bool:
            t = t.strip().lower()
            return "?" in t or any(t.startswith(w) for w in
                ("what", "who", "where", "when", "how", "which", "is my", "do you", "can you", "tell me"))

        if not _is_question(user_text):
            try:
                from clsplusplus.models import WriteRequest as _WriteReq
                await memory_service.write(
                    _WriteReq(text=user_text, namespace=ns, source="user", salience=0.8),
                    trace_id=tid,
                )
            except Exception:
                pass

        # 2. Search memory for relevant context
        memory_hits = []
        try:
            from clsplusplus.models import ReadRequest as _ReadReq
            read_resp = await memory_service.read(
                _ReadReq(query=user_text, namespace=ns, limit=8),
                trace_id=tid,
            )
            memory_hits = [i.text for i in (read_resp.items or [])]
        except Exception:
            pass

        memory_used = len(memory_hits) > 0
        memory_count = len(memory_hits)

        # 3. Build conversation history (last 6 turns)
        history_lines = []
        for m in sess.messages[-6:]:
            prefix = "User" if m.role == "user" else "Assistant"
            history_lines.append(f"{prefix}: {m.content}")
        conv_block = "\n".join(history_lines)
        history_line_count = len(history_lines)

        # 4. Build augmented system prompt
        mem_block = ("\n".join(f"- {t}" for t in memory_hits)
                     if memory_hits else "No prior memory context.")
        augmented_prompt = (
            "You are a helpful, friendly assistant with persistent memory.\n\n"
            f"Relevant memory context:\n{mem_block}\n\n"
            + (f"Conversation so far:\n{conv_block}\n\n" if conv_block else "")
            + "Answer naturally. Use memory context only when relevant."
        )

        # 5. Call LLM
        reply = "No LLM configured."
        try:
            if model == "claude":
                reply = await call_claude(settings, augmented_prompt, user_text)
            elif model == "openai":
                reply = await call_openai(settings, augmented_prompt, user_text)
            elif model == "gemini":
                reply = await call_gemini(settings, augmented_prompt, user_text)
        except Exception as exc:
            reply = f"LLM error: {exc}"

        # 6. Store assistant reply to memory
        try:
            from clsplusplus.models import WriteRequest as _WriteReq
            await memory_service.write(
                _WriteReq(text=f"Assistant replied: {reply[:400]}",
                          namespace=ns, source=f"assistant.{model}", salience=0.6),
                trace_id=tid,
            )
        except Exception:
            pass

        # 7. Persist messages in session
        sess.messages.append(_ChatMsg(role="user", content=user_text))
        sess.messages.append(_ChatMsg(
            role="assistant", content=reply,
            memory_used=memory_used, memory_count=memory_count,
        ))

        # 8. Build debug snapshot
        debug_info = {
            "model_used": model,
            "user_message": user_text,
            "memory_searched": memory_hits,
            "conversation_history_lines": history_line_count,
            "augmented_prompt": augmented_prompt,
            "phase_dynamics": _phase_dynamics(ns),
        }

        return {
            "reply": reply,
            "memory_used": memory_used,
            "memory_count": memory_count,
            "debug": debug_info,
        }

    # =========================================================================
    # Trace / Call Graph API
    # =========================================================================

    @app.get("/v1/trace/{trace_id}")
    async def get_trace(trace_id: str):
        """Return the full call graph tree for a trace UUID."""
        trace = tracer.get(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found (max {tracer.MAX_TRACES} traces kept in memory)")
        return trace.to_dict()

    @app.get("/v1/traces")
    async def list_traces(limit: int = Query(default=50, ge=1, le=200)):
        """List recent traces (newest first)."""
        return {"traces": tracer.list_recent(limit)}

    @app.get("/v1/memory/namespaces")
    async def list_namespaces():
        """Return all namespaces that have items in the PhaseMemoryEngine."""
        ns_list = []
        for ns, items in memory_service.engine._items.items():
            if not items:
                continue
            floor = memory_service.engine.STRENGTH_FLOOR
            phases = {"gas": 0, "liquid": 0, "solid": 0, "glass": 0}
            for item in items:
                d = item.to_debug_dict(strength_floor=floor)
                phases[d["phase"]] += 1
            # most-recently written item
            latest = max(items, key=lambda i: i.birth_order)
            ns_list.append({
                "namespace": ns,
                "total": len(items),
                "phases": phases,
                "latest_text": (latest.fact.raw_text or "")[:60],
                "latest_birth_order": latest.birth_order,
            })
        # Sort by most recently active first
        ns_list.sort(key=lambda x: x["latest_birth_order"], reverse=True)
        return {"namespaces": ns_list}

    @app.get("/v1/memory/phases")
    async def memory_phases(namespace: str = Query(default="default")):
        """Return items grouped by thermodynamic phase (gas/liquid/solid/glass).

        Used by the live Memory Phase visualiser in the Trace UI.
        Returns the 30 most recent items per phase, sorted by birth_order desc.
        """
        validate_namespace(namespace)
        raw_items = memory_service.engine._items.get(namespace, [])
        floor = memory_service.engine.STRENGTH_FLOOR

        phases: dict[str, list] = {"gas": [], "liquid": [], "solid": [], "glass": []}
        for item in raw_items:
            d = item.to_debug_dict(strength_floor=floor)
            # Attach indexed tokens (first 12, human-readable)
            d["tokens"] = item.indexed_tokens[:12]
            # Attach schema absorption count if solid/glass
            if item.schema_meta is not None:
                d["absorbed_count"] = len(item.schema_meta.H_history)
            else:
                d["absorbed_count"] = 0
            phases[d["phase"]].append(d)

        # Sort each phase by birth_order descending (most recent first)
        max_birth = max((x["birth_order"] for v in phases.values() for x in v), default=0)
        for phase_items in phases.values():
            phase_items.sort(key=lambda x: x["birth_order"], reverse=True)

        return {
            "namespace": namespace,
            "max_birth_order": max_birth,
            "total": len(raw_items),
            "phases": {
                p: {"items": items[:30], "total": len(items)}
                for p, items in phases.items()
            },
        }

    _api_logger = logging.getLogger(__name__)

    @app.on_event("startup")
    async def startup():
        """Start background hippocampal replay loop — fires every 5 minutes per active namespace."""
        async def _periodic_replay():
            while True:
                await asyncio.sleep(300)  # 5 minutes
                for ns in list(memory_service.engine._items.keys()):
                    try:
                        rehearsed = memory_service.engine.recall_long_tail(ns, batch_size=50)
                        if rehearsed:
                            _api_logger.debug(
                                "Periodic recall_long_tail ns=%s rehearsed=%d", ns, rehearsed
                            )
                    except Exception:
                        pass

        asyncio.create_task(_periodic_replay())

    @app.on_event("shutdown")
    async def shutdown():
        """Cleanly close all connection pools on shutdown."""
        for store in [memory_service.l1, memory_service.l2]:
            if hasattr(store, "close"):
                try:
                    await store.close()
                except Exception:
                    pass
        try:
            await integration_service.close()
        except Exception:
            pass

    # Serve website static files if the directory exists
    if _website_dir and FilePath(_website_dir).is_dir():
        app.mount("/", StaticFiles(directory=_website_dir, html=True), name="website")

    return app


app = create_app()

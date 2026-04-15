"""CLS++ REST API - FastAPI application."""

import asyncio
import collections
import logging
import time as _time
import uuid as _uuid_mod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

import os
import random
from pathlib import Path as FilePath

from fastapi import FastAPI, HTTPException, Path, Query, Request, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from clsplusplus.config import Settings
from clsplusplus.integration_service import IntegrationService
from clsplusplus.memory_service import MemoryService
from clsplusplus.user_service import UserService
from clsplusplus.middleware import AuthMiddleware, QuotaMiddleware, RateLimitMiddleware, RequestIdMiddleware, TracingMiddleware
from clsplusplus.tracer import tracer
from clsplusplus.models import (
    AdjudicateRequest,
    ApiKeyCreate,
    DemoChatRequest,
    ForgetRequest,
    HealthResponse,
    IntegrationCreate,
    MemoryCycleRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PromptIngestRequest,
    RazorpayVerifyRequest,
    ReadRequest,
    ReadResponse,
    TierUpgradeRequest,
    UserLoginRequest,
    UserProfileUpdateRequest,
    UserRegisterRequest,
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
    from clsplusplus.stores.integration_store import IntegrationStore
    _integration_store = IntegrationStore(settings)
    integration_service = IntegrationService(settings, store=_integration_store)
    user_service = UserService(settings)
    logger = logging.getLogger(__name__)

    # Launch-mode waitlist service (rate-limited rollout + social-proof widget)
    from clsplusplus.waitlist_service import WaitlistService, WaitlistError
    waitlist_service = WaitlistService(
        settings,
        user_store=user_service.store,
        email_service=user_service.email,
    )

    # Context log for authenticated API reads (Claude Code hooks, etc.)
    from collections import defaultdict
    _api_context_log: dict[str, collections.deque] = defaultdict(lambda: collections.deque(maxlen=100))

    # ── Topical Resonance Graph — cross-LLM session coupling ──
    from clsplusplus.topical_resonance import TopicalResonanceGraph
    _trg = TopicalResonanceGraph(engine=memory_service.engine)

    # ── Prompt Log & Context Log — persistent stores (Tier 2) ──
    from clsplusplus.prompt_log import PromptLogStore, ContextLogStore
    _prompt_log = PromptLogStore(settings)
    _context_log_store = ContextLogStore(settings)

    # ── Namespace Resolver — canonical namespace unification ──
    from clsplusplus.namespace_resolver import NamespaceResolver
    _namespace_resolver = NamespaceResolver(settings)

    app = FastAPI(
        title="CLS++ API",
        description="Brain-inspired, model-agnostic persistent memory for LLMs",
        version="7.0.0",
        docs_url=None,
        redoc_url=None,
    )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui():
        swagger_html = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " — API Documentation",
            swagger_css_url="/swagger-override.css",
        )
        # Inject video background after <body>
        body = swagger_html.body.decode()
        video_html = (
            '<div class="video-bg" style="position:fixed;inset:0;z-index:-2;overflow:hidden">'
            '<video autoplay muted loop playsinline poster="/beach-bg.jpg" '
            'style="width:100%;height:100%;object-fit:cover">'
            '<source src="/sunrise-ocean.mp4" type="video/mp4"></video></div>'
        )
        body = body.replace("<body>", "<body>" + video_html, 1)
        return HTMLResponse(content=body)

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc():
        return get_redoc_html(
            openapi_url=app.openapi_url,
            title=app.title + " — API Documentation",
        )

    # Explicit allowed origins — configurable via CLS_CORS_ORIGINS env var (comma-separated).
    # Chrome extension origins use the literal "chrome-extension://*" pattern which the
    # CORSMiddleware regex_origins list handles; the explicit list covers the web properties.
    _default_origins = "https://www.clsplusplus.com,https://clsplusplus.com"
    _cors_env = os.environ.get("CLS_CORS_ORIGINS", _default_origins)
    cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
    cors_regex = r"^chrome-extension://[a-z]{32}$"  # any Chrome extension
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["Content-Type", "Authorization", "X-Request-Id", "X-Trace-Id", "X-Session-Id"],
        expose_headers=["X-Request-Id", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )
    # Middleware execution order: outermost (added last) runs first.
    # TracingMiddleware → RequestId → RateLimit → Auth → Quota → route handler
    app.add_middleware(QuotaMiddleware, settings=settings)
    app.add_middleware(AuthMiddleware, settings=settings, integration_store=_integration_store)
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

    def _resolve_namespace(req, request: Request):
        """Override namespace with the API key's integration namespace when default."""
        if req.namespace == "default":
            ns = getattr(request.state, "namespace", None)
            if ns:
                req.namespace = ns

    # Detect website directory (in Docker: /app/website, local dev: ../website relative to src)
    _website_dir = os.environ.get("CLS_WEBSITE_DIR")
    if not _website_dir:
        _candidate = FilePath(__file__).resolve().parent.parent.parent / "website"
        if _candidate.is_dir():
            _website_dir = str(_candidate)

    # A/B/C landing page variant mapping
    _LANDING_VARIANTS = {
        "A": "index.html",        # Current design
        "B": "landing-d.html",    # Command Center (3-column app)
        "C": "landing-e.html",    # Ambient Focus (minimal, ambient)
    }
    _VARIANT_KEYS = list(_LANDING_VARIANTS.keys())

    @app.get("/")
    async def root(request: Request, variant: Optional[str] = Query(None)):
        """Serve landing page with A/B/C variant assignment."""
        if not _website_dir:
            return {
                "name": "CLS++ API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/v1/memory/health",
            }

        # 1. Admin override via ?variant=B
        chosen = None
        if variant and variant.upper() in _LANDING_VARIANTS:
            chosen = variant.upper()

        # 2. Check existing cookie
        if not chosen:
            chosen = request.cookies.get("cls_variant")
            if chosen not in _LANDING_VARIANTS:
                chosen = None

        # 3. Random assignment for new visitors
        if not chosen:
            chosen = random.choice(_VARIANT_KEYS)

        # Serve the variant file
        page_file = FilePath(_website_dir) / _LANDING_VARIANTS[chosen]
        if not page_file.exists():
            # Fallback to index.html if variant file missing
            page_file = FilePath(_website_dir) / "index.html"
            chosen = "A"

        from starlette.responses import Response
        with open(str(page_file), "r") as f:
            html_content = f.read()
        response = Response(content=html_content, media_type="text/html")

        # Set variant cookie (30 days)
        response.set_cookie(
            key="cls_variant",
            value=chosen,
            max_age=30 * 24 * 60 * 60,
            path="/",
            samesite="lax",
        )
        return response

    @app.get("/health")
    async def health_check():
        """Quick health check for Render/load balancers — must return 200 fast."""
        return {"status": "ok", "version": "7.0.0"}

    @app.get("/v1/health")
    async def v1_health():
        """Quick liveness probe. chat.js and other clients call this on startup."""
        return {"status": "ok", "version": getattr(settings, "version", "0.7")}

    @app.get("/v1/config/analytics")
    async def analytics_config():
        """Return PostHog API key from environment. No auth required."""
        return {"posthog_key": os.environ.get("POSTHOG_API_KEY", "")}

    @app.get("/v1/config/analytics-dashboard")
    async def analytics_dashboard_config():
        """Return PostHog shared dashboard URL from environment. No auth required (admin-only page)."""
        return {"dashboard_url": os.environ.get("POSTHOG_DASHBOARD_URL", "")}

    # Per-user metrics emitter (shared across all endpoints)
    from clsplusplus.metrics import MetricsEmitter
    _metrics = MetricsEmitter(settings)
    memory_service._metrics = _metrics  # Wire metrics into memory service

    async def _record_usage(operation: str, request: Request):
        """Fire-and-forget usage tracking. Must never crash a user request."""
        try:
            api_key = getattr(request.state, "api_key", None)
            if api_key and settings.track_usage:
                from clsplusplus.usage import record_usage, record_operation
                await record_usage(api_key, operation, settings)
                await record_operation(api_key, settings)
            # Per-user metrics (emit even without track_usage for admin visibility)
            user_id = getattr(request.state, "user_id", None)
            if user_id:
                await _metrics.emit(user_id, operation)
        except Exception:
            pass  # Usage tracking failure must not affect user responses

    @app.post("/v1/memory/write")
    async def write_memory(req: WriteRequest, request: Request):
        """Write memory. Flows to L0, promotes to L1 if score warrants."""
        _resolve_namespace(req, request)
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
        _resolve_namespace(req, request)
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
        """Read memories by semantic query across all stores + TRG cross-session.

        Cascade recall:
          1. TRG cross-session prompts (topically filtered, <0.1ms)
          2. PhaseMemoryEngine + L1/L2/L3 (thermodynamic retrieval, <1ms hot)
          3. Phase-weighted merge
        """
        _resolve_namespace(req, request)
        tid = _trace_id(request)
        _t0 = _time.monotonic()

        with tracer.span(tid, "api.read", "api",
                         input=req.query[:200],
                         namespace=req.namespace, limit=req.limit) as api_hop:

            # ── Layer 1: TRG cross-session recall (topically gated) ──
            session_id = request.headers.get("X-Session-Id", "")
            cross_items = []
            if session_id:
                from clsplusplus.topical_resonance import PromptEntry as TRGPromptEntry
                cross_results = _trg.recall_cross_session(
                    session_id, req.query, req.namespace, limit=5)
                for score, entry in cross_results:
                    from clsplusplus.models import MemoryItem, StoreLevel
                    cross_items.append(MemoryItem(
                        id=f"trg-{entry.session_id[:8]}-{entry.sequence_num}",
                        text=entry.content[:500],
                        namespace=req.namespace,
                        store_level=StoreLevel.L0,
                        source=entry.llm_provider,
                        timestamp=datetime.utcfromtimestamp(entry.timestamp),
                        confidence=min(1.0, score),
                        subject=f"cross-session:{entry.llm_provider}",
                    ))

            # ── Layer 2: Deep memory (PhaseMemoryEngine + L1/L2/L3) ──
            result = await memory_service.read(req, trace_id=tid)
            items = result.items or []

            # ── Layer 2b: Always supplement with L1 kNN to catch facts the
            # engine consolidated away. Short personal facts like "dingu comes
            # on her birthday" get merged by the engine's dedup but L1 keeps
            # the original verbatim text. ──
            try:
                query_emb = memory_service.embedding_service.embed(req.query)
                l1_items = await memory_service.l1.read(
                    query_emb, req.namespace, limit=req.limit)
                existing_ids = {i.id for i in items}
                for l1_item in (l1_items or []):
                    if l1_item.id not in existing_ids:
                        items.append(l1_item)
                        existing_ids.add(l1_item.id)
                result.items = items
            except Exception:
                pass  # L1 supplement is best-effort

            # ── Quality filter: prioritize real facts over garbage schemas ──
            # L2 schemas like "[Schema: raj] raj property apartment house" are
            # token-soup abstractions. They help TRR scoring but are useless
            # as injected context. Prioritize L1 episodic memories (actual sentences)
            # and only include L2 schemas if they contain real readable content.
            real_facts = []
            questions = []
            schema_filler = []
            for item in items:
                text = item.text or ""
                is_schema = text.startswith("[Schema:")
                is_question = text.rstrip().endswith("?")
                if is_schema:
                    words = text.split("]", 1)[-1].strip().split()
                    has_structure = any(len(w) > 5 for w in words) and len(words) > 3
                    if has_structure:
                        schema_filler.append(item)
                elif is_question:
                    # Questions are low-value as injected context — demote
                    questions.append(item)
                else:
                    real_facts.append(item)
            # Priority: real statements > schemas > questions
            items = real_facts + schema_filler + questions
            result.items = items[:req.limit]
            items = result.items

            # ── Layer 3: Merge — cross-session first (fresh), then deep ──
            if cross_items:
                cross_limit = min(len(cross_items), max(2, req.limit * 4 // 10))
                deep_limit = req.limit - cross_limit
                merged = cross_items[:cross_limit] + items[:deep_limit]
                result.items = merged[:req.limit]
                items = result.items

                # ── Promotion bridge: reinforce engine items via cross-session ──
                # Feeds back into s(t) = exp(-Δt/τ) × (1 + β·ln(1+R))
                # driving Gas→Liquid→Solid→Glass promotion pipeline
                _trg.reinforce_cross_session(session_id, req.query, req.namespace)

            preview = items[0].text[:80] if items else "no results"
            tracer.add_metadata(tid, api_hop,
                                output=f"{len(items)} items (cross={len(cross_items)}): {preview}")

        await _record_usage("read", request)
        result.trace_id = tid
        latency_ms = int((_time.monotonic() - _t0) * 1000)

        # Log to context log (in-memory for backward compatibility + persistent)
        if items:
            source = request.headers.get("User-Agent", "")
            model = "claude-code" if "curl" in source.lower() or not source else req.namespace
            _api_context_log[req.namespace].append({
                "model": model,
                "query": req.query[:120],
                "memories_sent": [i.text for i in items],
                "count": len(items),
                "ts": datetime.utcnow().isoformat(),
            })

            # Persistent context log (fire-and-forget)
            user_id = getattr(request.state, "user_id", None) or "anonymous"
            asyncio.create_task(_context_log_store.append(
                user_id=str(user_id),
                namespace=req.namespace,
                session_id=session_id,
                llm_provider=model,
                query=req.query[:500],
                memories_sent=[i.text for i in items[:20]],
                memory_ids=[i.id for i in items[:20]],
                memory_count=len(items),
                latency_ms=latency_ms,
            ))

        return result

    @app.post("/v1/memories/retrieve", response_model=ReadResponse)
    async def retrieve_memories(req: ReadRequest, request: Request):
        """Product alias: POST /memories/retrieve -> read."""
        _resolve_namespace(req, request)
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

    @app.get("/v1/context-log")
    async def get_api_context_log(request: Request):
        """Return context injection history for Claude Code and API consumers."""
        ns = getattr(request.state, "namespace", None) or "default"
        return {
            "namespace": ns,
            "log": list(reversed(_api_context_log.get(ns, []))),
        }

    @app.get("/v1/memory/personal")
    async def get_personal_facts(request: Request, limit: int = Query(20, ge=1, le=100)):
        """Return personal facts only — pre-filtered for Custom Instructions sync.
        Excludes questions, dev commands, short messages, schema tokens."""
        ns = getattr(request.state, "namespace", None) or "default"
        await memory_service.ensure_loaded(ns)
        engine = memory_service.engine
        items = engine._items.get(ns, [])

        # Get alive items sorted by birth_order (recent first)
        alive = [i for i in items if i.consolidation_strength >= engine.STRENGTH_FLOOR]
        alive.sort(key=lambda i: i.birth_order, reverse=True)

        facts = []
        for i in alive:
            t = i.fact.raw_text
            if len(t) < 8 or len(t) > 300:
                continue
            if t.startswith('[') or t.endswith('?'):
                continue
            if t.startswith('Stop hook') or t.startswith('You said'):
                continue
            facts.append({"text": t, "id": i.id})
            if len(facts) >= limit:
                break

        # Also supplement from L1 PostgreSQL
        try:
            pool = await memory_service.l1.get_pool()
            rows = await pool.fetch("""
                SELECT id, text FROM l1_memories
                WHERE namespace = $1
                  AND length(text) > 8 AND length(text) < 300
                  AND text NOT LIKE '[%' AND text NOT LIKE '%?'
                  AND text NOT LIKE 'Stop hook%' AND text NOT LIKE 'You said%'
                ORDER BY created_at DESC LIMIT $2
            """, ns, limit)
            existing = {f["id"] for f in facts}
            for r in rows:
                if str(r["id"]) not in existing and len(facts) < limit:
                    facts.append({"text": r["text"], "id": str(r["id"])})
        except Exception:
            pass

        return {"facts": facts, "count": len(facts), "namespace": ns}

    # ── Prompt Ingestion — Cross-LLM Context Pipeline ─────────────────

    @app.post("/v1/prompts/ingest")
    async def ingest_prompts(body: PromptIngestRequest, request: Request):
        """Batch ingest prompts from an LLM session.

        Three-tier pipeline:
          Tier 0: TopicalResonanceGraph (in-process, <0.1ms)
          Tier 1: PhaseMemoryEngine fact extraction (in-process, <0.5ms)
          Tier 2: prompt_log PostgreSQL (fire-and-forget, async)
        """
        _resolve_namespace(body, request)
        ns = body.namespace
        user_id = getattr(request.state, "user_id", None) or "anonymous"

        for entry in body.entries:
            content = entry.content.strip()
            if not content:
                continue

            # Tier 0: TRG — update session oscillator + cross-session coupling
            _trg.on_prompt(
                session_id=body.session_id,
                content=content,
                llm_provider=body.llm_provider,
                namespace=ns,
                role=entry.role,
                sequence_num=entry.sequence_num,
            )

            # Tier 1: Extract facts for deep memory (user messages only, >=10 chars)
            if entry.role == "user" and len(content) >= 10:
                asyncio.create_task(memory_service.write(
                    WriteRequest(
                        text=content[:1000],
                        namespace=ns,
                        source=body.llm_provider,
                        metadata={"session_id": body.session_id},
                    ),
                ))

        # Tier 2: Persist to PostgreSQL (fire-and-forget)
        asyncio.create_task(_prompt_log.batch_append(
            user_id=str(user_id),
            namespace=ns,
            session_id=body.session_id,
            llm_provider=body.llm_provider,
            llm_model=body.llm_model or "",
            client_type=body.client_type,
            entries=[{"role": e.role, "content": e.content,
                      "sequence_num": e.sequence_num,
                      "metadata": e.metadata} for e in body.entries],
        ))

        await _record_usage("ingest", request)
        return {"ok": True, "count": len(body.entries)}

    @app.get("/v1/prompts/sessions")
    async def list_prompt_sessions(request: Request, limit: int = Query(20, ge=1, le=100)):
        """List recent LLM sessions for the authenticated user."""
        ns = getattr(request.state, "namespace", None) or "default"
        sessions = await _prompt_log.get_user_sessions_by_namespace(ns, limit)
        # Serialize datetime fields
        for s in sessions:
            for k in ("started_at", "last_at"):
                if k in s and s[k]:
                    s[k] = s[k].isoformat()
        return {"sessions": sessions, "namespace": ns}

    @app.get("/v1/prompts/sessions/{session_id}")
    async def get_prompt_session(session_id: str, limit: int = Query(100, ge=1, le=500)):
        """Get full conversation for a session."""
        messages = await _prompt_log.get_session(session_id, limit)
        for m in messages:
            if "created_at" in m and m["created_at"]:
                m["created_at"] = m["created_at"].isoformat()
            if "user_id" in m:
                m["user_id"] = str(m["user_id"])
            if "id" in m:
                m["id"] = str(m["id"])
        return {"session_id": session_id, "messages": messages}

    @app.get("/v1/prompts/timeline")
    async def get_prompt_timeline(request: Request,
                                  limit: int = Query(50, ge=1, le=200)):
        """Unified timeline of all prompts across sessions."""
        ns = getattr(request.state, "namespace", None) or "default"
        entries = await _prompt_log.get_timeline(ns, limit)
        for e in entries:
            if "created_at" in e and e["created_at"]:
                e["created_at"] = e["created_at"].isoformat()
            if "id" in e:
                e["id"] = str(e["id"])
        return {"entries": entries, "namespace": ns}

    @app.get("/v1/trg/state")
    async def get_trg_state(request: Request):
        """Debug endpoint: current Topical Resonance Graph state."""
        return _trg.debug_state()

    @app.get("/v1/memory/list")
    async def list_memories(request: Request, limit: int = Query(100, ge=1, le=500), namespace: str = Query("")):
        """List all memories in the authenticated namespace (no semantic search)."""
        ns = namespace or getattr(request.state, "namespace", None) or "default"
        await memory_service.ensure_loaded(ns)
        engine = memory_service.engine
        items = engine._items.get(ns, [])
        # Filter alive items, sort by birth_order (most recent first)
        alive = [i for i in items if i.consolidation_strength >= engine.STRENGTH_FLOOR]
        alive.sort(key=lambda i: i.birth_order, reverse=True)
        result_items = []
        for i in alive[:limit]:
            raw = i.fact.raw_text
            # Detect Claude Code hook source from text prefix
            source = "claude-code-hook" if raw.startswith("[User prompt]") or raw.startswith("[Assistant]") else "user"
            result_items.append({
                "id": i.id,
                "text": raw,
                "source": source,
                "namespace": i.namespace,
                "store_level": "L1",
                "timestamp": str(i.event_at or ""),
                "confidence": round(i.consolidation_strength, 3),
                "metadata": {},
            })
        return {
            "items": result_items,
            "count": len(result_items),
            "total": len(items),
            "namespace": ns,
        }

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
    async def prewarm_namespace(request: Request, namespace: str = _ns_query()):
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
        await _record_usage("prewarm", request)
        return {"status": "loaded" if already else "loading", "namespace": namespace}

    @app.post("/v1/memory/sleep")
    async def trigger_sleep(request: Request, namespace: str = _ns_query()):
        """Trigger nightly sleep cycle (admin)."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        report = await sleep_orchestrator.run(namespace)
        await _record_usage("consolidation", request)
        return report

    @app.post("/v1/memories/consolidate")
    async def consolidate_memories(request: Request, namespace: str = _ns_query()):
        """Product alias: POST /memories/consolidate -> sleep."""
        try:
            validate_namespace(namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        report = await sleep_orchestrator.run(namespace)
        await _record_usage("consolidation", request)
        return report

    @app.delete("/v1/memory/forget")
    async def forget_memory(req: ForgetRequest, request: Request):
        """Delete a memory by ID (RTBF)."""
        _resolve_namespace(req, request)
        deleted = await memory_service.delete(req.item_id, req.namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        await _record_usage("delete", request)
        return {"deleted": True, "item_id": req.item_id}

    @app.delete("/v1/memories/forget")
    async def forget_memory_alias(req: ForgetRequest, request: Request):
        """Product alias: DELETE /memories/forget."""
        _resolve_namespace(req, request)
        deleted = await memory_service.delete(req.item_id, req.namespace)
        if not deleted:
            raise HTTPException(status_code=404, detail="Item not found")
        await _record_usage("delete", request)
        return {"deleted": True, "item_id": req.item_id}

    @app.post("/v1/memory/adjudicate_conflict")
    async def adjudicate_conflict(req: AdjudicateRequest, request: Request):
        """Submit conflicting fact + evidence for reconsolidation gate."""
        result = await memory_service.adjudicate(
            new_text=req.new_fact,
            namespace=req.namespace,
            evidence=req.evidence,
            existing_item_id=req.existing_item_id,
        )
        # If the returned item is the same as the existing one, quorum was not met
        decision = "rejected" if (req.existing_item_id and result.id == req.existing_item_id) else "accepted"
        await _record_usage("adjudication", request)
        return {"decision": decision, "new_id": result.id}

    @app.get("/v1/demo/status")
    async def demo_status():
        """Check which LLM keys are configured (for debugging)."""
        return {
            "claude": bool(getattr(settings, "anthropic_api_key", None)),
            "openai": bool(getattr(settings, "openai_api_key", None)),
            "gemini": bool(getattr(settings, "google_api_key", None)),
        }

    @app.post("/v1/demo/classify")
    async def demo_classify(request: Request):
        """Classify visitor intent for generative homepage.

        Uses local keyword matching first, falls back to Claude API.
        Returns { persona: "developer"|"founder"|"researcher", subtitle: "..." }
        """
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text or len(text) < 2:
            return {"persona": "developer", "subtitle": ""}

        # Local classifier (instant, no API call)
        lower = text.lower()
        dev_signals = ["api", "sdk", "integrate", "code", "build", "install", "npm", "pip",
                       "docker", "deploy", "endpoint", "webhook", "python", "javascript",
                       "fastapi", "langchain", "crewai", "framework", "library", "package"]
        founder_signals = ["startup", "product", "saas", "users", "customers", "ship", "launch",
                           "mvp", "pricing", "scale", "revenue", "business", "app", "platform",
                           "chatbot", "support bot", "ai product", "remember"]
        research_signals = ["paper", "neuroscience", "cls", "hippocampus", "neocortex",
                            "consolidation", "mcclelland", "memory theory", "cognitive",
                            "research", "academic", "architecture", "dual store", "schema", "phase"]

        dev = sum(1 for s in dev_signals if s in lower)
        founder = sum(1 for s in founder_signals if s in lower)
        research = sum(1 for s in research_signals if s in lower)

        if research > dev and research > founder:
            return {"persona": "researcher", "subtitle": ""}
        if founder > dev:
            return {"persona": "founder", "subtitle": ""}
        if dev > 0:
            return {"persona": "developer", "subtitle": ""}

        # Ambiguous — try Claude API for nuanced classification
        if settings.anthropic_api_key:
            try:
                import httpx, json as _json
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.post("https://api.anthropic.com/v1/messages", headers={
                        "x-api-key": settings.anthropic_api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }, json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 200,
                        "messages": [{"role": "user", "content": f'Classify this visitor intent into exactly ONE persona. Visitor typed: "{text}"\n\nRespond ONLY in JSON: {{"persona": "developer|founder|researcher", "subtitle": "One tailored sentence"}}'}],
                    })
                    if r.status_code == 200:
                        content = r.json().get("content", [{}])[0].get("text", "")
                        parsed = _json.loads(content.replace("```json", "").replace("```", "").strip())
                        return {"persona": parsed.get("persona", "developer"), "subtitle": parsed.get("subtitle", "")}
            except Exception as e:
                logger.warning("Demo classify Claude fallback failed: %s", e)

        return {"persona": "developer", "subtitle": ""}

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
        request: Request,
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
        await _record_usage("delete", request)
        return {"deleted": True, "item_id": item_id}

    @app.delete("/v1/memory/wipe")
    async def wipe_all_memories(request: Request, confirm: bool = Query(False)):
        """Nuclear option: delete ALL memories for the authenticated namespace.
        Clears engine, L1, L2, L3. Requires confirm=true query parameter."""
        if not confirm:
            raise HTTPException(
                status_code=400,
                detail="Destructive operation requires confirm=true query parameter. "
                       "Example: DELETE /v1/memory/wipe?confirm=true",
            )
        ns = getattr(request.state, "namespace", None) or "default"
        # Clear engine
        memory_service.engine._items.pop(ns, None)
        memory_service.engine._token_index.pop(ns, None)
        # Clear L1
        try:
            pool = await memory_service.l1.get_pool()
            await pool.execute("DELETE FROM l1_memories WHERE namespace = $1", ns)
        except Exception:
            pass
        # Clear L2
        try:
            pool = await memory_service.l2.get_pool()
            await pool.execute("DELETE FROM l2_nodes WHERE namespace = $1", ns)
        except Exception:
            pass
        # Clear L3
        try:
            pool = await memory_service.l3.get_pool()
            await pool.execute("DELETE FROM l3_engrams WHERE namespace = $1", ns)
        except Exception:
            pass
        await _record_usage("wipe", request)
        return {"wiped": True, "namespace": ns}

    @app.get("/v1/usage")
    async def usage_endpoint(request: Request):
        """Usage metrics for current period with tier info. Requires API key when auth enabled."""
        from clsplusplus.auth import get_api_key_from_request
        api_key = getattr(request.state, "api_key", None) or get_api_key_from_request(request.headers.get("Authorization"))
        if settings.require_api_key and not api_key:
            raise HTTPException(status_code=401, detail="API key required")
        from clsplusplus.tiers import get_tier, get_quota_status
        tier = get_tier(settings)
        return await get_quota_status(api_key or "anonymous", tier, settings)

    @app.get("/v1/billing/usage")
    async def billing_usage(request: Request):
        """Billing API: usage for current period (alias for /v1/usage)."""
        return await usage_endpoint(request)

    # =========================================================================
    # User Auth API — Registration, login, Google OAuth, profile
    # =========================================================================

    def _set_session_cookie(response: JSONResponse, token: str) -> JSONResponse:
        response.set_cookie(
            key="cls_session",
            value=token,
            httponly=True,
            secure=settings.cookie_secure,
            samesite="lax",
            max_age=7 * 86400,  # 7 days
            path="/",
        )
        return response

    @app.post("/v1/auth/register")
    async def register_user(request: Request, req: UserRegisterRequest):
        """Step 1: Validate email/password, send OTP. Does NOT create user yet.

        During the rate-limited launch, walk-in signups are blocked once the
        user count hits `max_active_users`. The client is told to use the
        waitlist widget instead. Waitlist-invited users bypass this via
        /v1/waitlist/accept which creates users directly.
        """
        try:
            exceeded, current, cap = await waitlist_service.is_launch_cap_exceeded()
            if exceeded:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "launch_cap_reached",
                        "message": (
                            "We're releasing CLS++ in small waves. "
                            "Join the waitlist and we'll email you the moment a seat opens up."
                        ),
                        "waitlist": True,
                        "current_users": current,
                        "cap": cap,
                    },
                )
        except Exception as e:
            logger.warning("Launch cap check failed (fail-open): %s", e)

        base_url = str(request.base_url).rstrip("/")
        if request.headers.get("x-forwarded-proto") == "https":
            base_url = base_url.replace("http://", "https://", 1)
        try:
            result = await user_service.register(req.email, req.password, req.name, base_url=base_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Registration error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Registration service unavailable")
        return JSONResponse(content=result)

    # =========================================================================
    # Launch waitlist — rate-limited rollout + social proof widget
    # =========================================================================

    @app.post("/v1/waitlist/join")
    async def waitlist_join(request: Request):
        """Public. Email → sends 6-digit OTP. Rate-limited per email internally."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        email = (body or {}).get("email", "")
        source = request.cookies.get("cls_variant", "")
        try:
            result = await waitlist_service.join(email, source_variant=source)
        except WaitlistError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Waitlist join error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Waitlist service unavailable")
        return JSONResponse(content=result)

    @app.post("/v1/waitlist/verify")
    async def waitlist_verify(request: Request):
        """Public. Email + OTP → enters the waiting queue, returns position."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        email = (body or {}).get("email", "")
        otp = (body or {}).get("otp_code", "")
        try:
            result = await waitlist_service.verify(email, otp)
        except WaitlistError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Waitlist verify error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Waitlist service unavailable")
        return JSONResponse(content=result)

    @app.get("/v1/waitlist/stats")
    async def waitlist_stats(email: Optional[str] = Query(None)):
        """Public. Live counters for the landing widget. Polled every 20s."""
        try:
            return JSONResponse(content=await waitlist_service.stats(email=email))
        except Exception as e:
            logger.warning("Waitlist stats error: %s", e)
            return JSONResponse(content={
                "waiting_count": getattr(settings, "waitlist_queue_seed_offset", 47),
                "active_now": getattr(settings, "waitlist_active_floor", 3),
            })

    @app.get("/v1/waitlist/accept")
    async def waitlist_accept(request: Request, token: str = Query("")):
        """Public. One-click activation: token → real user + JWT cookie → dashboard.

        This is the "2-hour onboarding window" landing point. Clicking the
        invite email link lands here; we create a real account with a random
        password, sign the user in, and redirect to the dashboard. They can set
        a real password from /profile.html afterwards.
        """
        if not token:
            raise HTTPException(status_code=400, detail="Missing invite token")
        try:
            user, jwt_token = await waitlist_service.activate_by_token(token)
        except WaitlistError as e:
            from starlette.responses import RedirectResponse
            return RedirectResponse(f"/?waitlist_error={str(e).replace(' ', '+')}")
        except Exception as e:
            logger.error("Waitlist activate error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Activation failed")
        from starlette.responses import RedirectResponse
        response = RedirectResponse("/dashboard.html?welcome=waitlist")
        return _set_session_cookie(response, jwt_token)

    @app.post("/v1/auth/verify-register")
    async def verify_register(request: Request):
        """Step 2: Verify OTP and create the actual user account."""
        body = await request.json()
        email = body.get("email", "").strip()
        otp_code = body.get("otp_code", "").strip()
        if not email or not otp_code or len(otp_code) != 6:
            raise HTTPException(status_code=400, detail="Email and 6-digit code required")
        try:
            user, token = await user_service.complete_registration(email, otp_code)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Verify-register error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Verification failed")
        response = JSONResponse(content=user)
        return _set_session_cookie(response, token)

    @app.get("/v1/auth/verify-register")
    async def verify_register_link(request: Request, token: str = Query("")):
        """Verify via magic link click and create user account."""
        if not token:
            raise HTTPException(status_code=400, detail="Missing token")
        try:
            user, jwt_token = await user_service.complete_registration_link(token)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Magic link register error: %s", e)
            raise HTTPException(status_code=500, detail="Verification failed")
        from starlette.responses import RedirectResponse
        response = RedirectResponse("/dashboard.html?verified=1")
        return _set_session_cookie(response, jwt_token)

    @app.post("/v1/auth/login")
    async def login_user(req: UserLoginRequest):
        """Login with email and password."""
        try:
            user, token = await user_service.login(req.email, req.password)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail="Login service unavailable")
        response = JSONResponse(content=user)
        return _set_session_cookie(response, token)

    @app.post("/v1/auth/forgot-password")
    async def forgot_password(request: Request, req: PasswordResetRequest):
        """Request a password reset token. Sends email if Resend is configured."""
        base_url = str(request.base_url).rstrip("/")
        if request.headers.get("x-forwarded-proto") == "https":
            base_url = base_url.replace("http://", "https://", 1)
        try:
            token = await user_service.request_password_reset(req.email, base_url=base_url)
        except Exception as e:
            logger.error("Password reset request error: %s", e)
            return {"detail": "If an account with that email exists, a reset link has been sent.", "token": None}
        if token:
            return {"detail": "Password reset link sent to your email. Valid for 1 hour.", "token": token}
        return {"detail": "If an account with that email exists, a reset link has been sent.", "token": None}

    @app.post("/v1/auth/reset-password")
    async def reset_password(req: PasswordResetConfirm):
        """Reset password using a valid token."""
        try:
            user = await user_service.reset_password(req.token, req.new_password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Password reset error: %s", e)
            raise HTTPException(status_code=500, detail="Password reset service unavailable")
        return {"detail": "Password has been reset successfully."}

    @app.post("/v1/auth/verify-email")
    async def verify_email_otp(request: Request):
        """Verify email using 6-digit OTP code (authenticated user)."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        body = await request.json()
        otp_code = body.get("otp_code", "").strip()
        if not otp_code or len(otp_code) != 6:
            raise HTTPException(status_code=400, detail="Invalid verification code format")
        try:
            await user_service.verify_email_otp(user_id, otp_code)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Email verification error: %s", e)
            raise HTTPException(status_code=500, detail="Verification service unavailable")
        return {"detail": "Email verified successfully."}

    @app.get("/v1/auth/verify-email")
    async def verify_email_link(request: Request, token: str = Query("")):
        """Verify email via magic link (public, redirects to dashboard)."""
        if not token:
            raise HTTPException(status_code=400, detail="Missing verification token")
        try:
            user_id = await user_service.verify_email_link(token)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Magic link verification error: %s", e)
            raise HTTPException(status_code=500, detail="Verification service unavailable")
        from starlette.responses import RedirectResponse
        return RedirectResponse("/dashboard.html?verified=1")

    @app.post("/v1/auth/resend-verification")
    async def resend_verification(request: Request):
        """Resend verification email for the current user."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        base_url = str(request.base_url).rstrip("/")
        if request.headers.get("x-forwarded-proto") == "https":
            base_url = base_url.replace("http://", "https://", 1)
        try:
            await user_service.resend_verification(user_id, base_url=base_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Resend verification error: %s", e)
            raise HTTPException(status_code=500, detail="Verification service unavailable")
        return {"detail": "Verification email sent."}

    @app.post("/v1/auth/logout")
    async def logout_user():
        """Clear session cookie."""
        response = JSONResponse(content={"detail": "Logged out"})
        response.delete_cookie("cls_session", path="/")
        return response

    @app.get("/v1/auth/me")
    async def get_current_user(request: Request):
        """Get current authenticated user from JWT cookie OR API key."""
        user_id = getattr(request.state, "user_id", None)

        # If API key auth (no user_id but has namespace), resolve user from integration
        if not user_id:
            ns = getattr(request.state, "namespace", None)
            api_key = getattr(request.state, "api_key", None)
            if ns and api_key:
                try:
                    # Resolve: namespace → integration → owner_email → user
                    integrations = await _integration_store.list_integrations(ns)
                    for intg in integrations:
                        owner_email = intg.get("owner_email")
                        if owner_email:
                            user = await user_service.store.get_by_email(owner_email)
                            if user:
                                return {
                                    "id": str(user.get("id", "")),
                                    "email": user.get("email", ""),
                                    "name": user.get("name", ""),
                                    "tier": user.get("tier", "free"),
                                    "is_admin": user.get("is_admin", False),
                                    "namespace": ns,
                                    "auth_method": "api_key",
                                }
                            break
                except Exception:
                    pass

        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            user = await user_service.get_user(user_id)
        except Exception:
            raise HTTPException(status_code=500, detail="User service unavailable")
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    @app.get("/v1/auth/google")
    async def google_auth_redirect(request: Request, redirect: str = "/dashboard.html"):
        """Redirect to Google OAuth consent screen."""
        from urllib.parse import urlencode
        if not settings.google_client_id:
            raise HTTPException(status_code=501, detail="Google OAuth not configured")
        callback_url = str(request.base_url).rstrip("/") + "/v1/auth/google/callback"
        # Force HTTPS when behind reverse proxy (Render, Cloudflare, etc.)
        if request.headers.get("x-forwarded-proto") == "https" or "onrender.com" in callback_url:
            callback_url = callback_url.replace("http://", "https://", 1)
        params = urlencode({
            "client_id": settings.google_client_id,
            "redirect_uri": callback_url,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "state": redirect,
        })
        from starlette.responses import RedirectResponse
        return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")

    @app.get("/v1/auth/google/callback")
    async def google_auth_callback(request: Request, code: str = "", state: str = "/dashboard.html"):
        """Handle Google OAuth callback — exchange code, create/login user, redirect."""
        if not code:
            raise HTTPException(status_code=400, detail="Missing authorization code")
        callback_url = str(request.base_url).rstrip("/") + "/v1/auth/google/callback"
        # Force HTTPS when behind reverse proxy (Render, Cloudflare, etc.)
        if request.headers.get("x-forwarded-proto") == "https" or "onrender.com" in callback_url:
            callback_url = callback_url.replace("http://", "https://", 1)
        try:
            user, token = await user_service.google_auth(code, callback_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="Google auth service unavailable")
        # Validate state is a safe relative path — reject open-redirect attempts
        _safe_redirect = "/dashboard.html"
        if state and state.startswith("/") and "://" not in state and not state.startswith("//"):
            _safe_redirect = state
        from starlette.responses import RedirectResponse
        response = RedirectResponse(_safe_redirect)
        _set_session_cookie(response, token)
        return response

    # =========================================================================
    # User Dashboard API — Per-user usage and tier management
    # =========================================================================

    @app.get("/v1/user/usage")
    async def user_usage(request: Request):
        """Usage metrics for the authenticated user."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        user = await user_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        from clsplusplus.tiers import Tier, get_quota_status
        tier = Tier(user["tier"])
        namespace = f"user-{user_id[:8]}"
        return await get_quota_status(namespace, tier, settings)

    @app.post("/v1/user/upgrade")
    async def upgrade_tier(req: TierUpgradeRequest, request: Request):
        """Change user tier (upgrade or downgrade)."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            user = await user_service.update_tier(user_id, req.tier)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="Upgrade service unavailable")
        return user

    @app.get("/v1/user/usage/history")
    async def user_usage_history(request: Request):
        """Usage history for the last 6 months for the authenticated user."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        namespace = f"user-{user_id[:8]}"
        from clsplusplus.usage import get_usage_history
        try:
            return await get_usage_history(namespace, months=6, settings=settings)
        except Exception:
            return []

    @app.get("/v1/user/integrations")
    async def user_integrations(request: Request):
        """List integrations for the authenticated user's namespace."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        namespace = f"user-{user_id[:8]}"
        try:
            integrations = await integration_service.list_all(namespace)
            return {"integrations": [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in integrations]}
        except Exception:
            return {"integrations": []}

    @app.patch("/v1/user/profile")
    async def update_profile(req: UserProfileUpdateRequest, request: Request):
        """Update user profile (name, email, password)."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            user = await user_service.update_profile(
                user_id=user_id,
                name=req.name,
                email=req.email,
                password=req.password,
                current_password=req.current_password,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="Profile update failed")
        return user

    # =========================================================================
    # Billing — Stripe Checkout & Customer Portal
    # =========================================================================

    @app.post("/v1/billing/checkout")
    async def billing_checkout(req: TierUpgradeRequest, request: Request):
        """Create a Stripe Checkout session for tier upgrade."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            from clsplusplus.stripe_service import create_checkout_session
            session_url = await create_checkout_session(
                user_id=user_id,
                tier=req.tier,
                settings=settings,
            )
            return {"url": session_url}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Stripe checkout error: %s", e)
            raise HTTPException(status_code=500, detail="Billing service unavailable")

    @app.get("/v1/billing/portal")
    async def billing_portal(request: Request):
        """Create a Stripe Customer Portal session for subscription management."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            from clsplusplus.stripe_service import create_portal_session
            portal_url = await create_portal_session(
                user_id=user_id,
                settings=settings,
            )
            return {"url": portal_url}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Stripe portal error: %s", e)
            raise HTTPException(status_code=500, detail="Billing service unavailable")

    @app.post("/v1/billing/webhook")
    async def billing_webhook(request: Request):
        """Handle Stripe webhook events."""
        payload = await request.body()
        sig = request.headers.get("stripe-signature", "")
        try:
            from clsplusplus.stripe_service import handle_webhook
            await handle_webhook(
                payload=payload,
                sig=sig,
                settings=settings,
                user_service=user_service,
            )
            return {"status": "ok"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Stripe webhook error: %s", e)
            raise HTTPException(status_code=500, detail="Webhook processing failed")

    # =========================================================================
    # Billing — Razorpay (active payment gateway)
    # =========================================================================

    @app.post("/v1/billing/order")
    async def billing_order(req: TierUpgradeRequest, request: Request):
        """Create a Razorpay Order for tier upgrade."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if req.tier == "free":
            raise HTTPException(status_code=400, detail="Cannot create order for the free tier")
        try:
            from clsplusplus.razorpay_service import create_order
            order = await create_order(
                user_id=user_id,
                tier=req.tier,
                settings=settings,
            )
            # Include prefill data for Razorpay checkout modal
            order["prefill"] = {
                "email": getattr(request.state, "user_email", ""),
            }
            return order
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Razorpay order error: %s", e)
            raise HTTPException(status_code=500, detail="Billing service unavailable")

    @app.post("/v1/billing/verify")
    async def billing_verify(req: RazorpayVerifyRequest, request: Request):
        """Verify Razorpay payment signature and upgrade user tier."""
        user_id = getattr(request.state, "user_id", None)
        if not user_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        try:
            from clsplusplus.razorpay_service import verify_payment
            await verify_payment(
                order_id=req.order_id,
                payment_id=req.payment_id,
                signature=req.signature,
                settings=settings,
                user_service=user_service,
                tier=req.tier,
                user_id=user_id,
            )
            return {"status": "ok", "tier": req.tier}
        except ValueError as e:
            # If signature is invalid, reject. But if already upgraded, that's OK.
            detail = str(e)
            if "invalid signature" in detail.lower():
                raise HTTPException(status_code=400, detail=detail)
            # Any other ValueError (e.g., user not found) — still return success
            # if user is already on the requested tier
            try:
                current = await user_service.get_user(user_id)
                if current and current.get("tier") == req.tier:
                    return {"status": "ok", "tier": req.tier}
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=detail)
        except Exception as e:
            logger.error("Razorpay verify error: %s", e)
            raise HTTPException(status_code=500, detail="Payment verification failed")

    @app.post("/v1/billing/razorpay-webhook")
    async def billing_razorpay_webhook(request: Request):
        """Handle Razorpay webhook events (backup verification)."""
        payload = await request.body()
        sig = request.headers.get("x-razorpay-signature", "")
        try:
            from clsplusplus.razorpay_service import handle_webhook as rp_handle_webhook
            await rp_handle_webhook(
                payload=payload,
                sig=sig,
                settings=settings,
                user_service=user_service,
            )
            return {"status": "ok"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Razorpay webhook error: %s", e)
            raise HTTPException(status_code=500, detail="Webhook processing failed")

    # =========================================================================
    # Admin Dashboard API — Protected by is_admin flag in JWT
    # =========================================================================

    @app.get("/admin/metrics/summary")
    async def admin_summary(request: Request):
        """Top-bar KPIs: Total Users, Revenue, Cost, Margin %."""
        _require_admin(request)
        try:
            from clsplusplus.tiers import Tier, TIER_PRICES
            from clsplusplus.cost_model import compute_cost

            tier_counts = await user_service.store.count_users_by_tier()
            total_users = sum(tier_counts.values())
            paying_users = total_users - tier_counts.get("free", 0)

            monthly_revenue = sum(
                tier_counts.get(t.value, 0) * TIER_PRICES[t]
                for t in Tier
            )

            aggregate = await _metrics.get_aggregate_metrics()
            monthly_cost = compute_cost(aggregate)
            margin = ((monthly_revenue - monthly_cost) / monthly_revenue * 100) if monthly_revenue > 0 else 0

            return {
                "total_users": total_users,
                "paying_users": paying_users,
                "free_users": tier_counts.get("free", 0),
                "monthly_revenue": round(monthly_revenue, 2),
                "monthly_cost": round(monthly_cost, 4),
                "margin_percent": round(margin, 1),
                "tier_counts": tier_counts,
            }
        except Exception as e:
            logger.error("Admin metrics error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Metrics unavailable")

    @app.get("/admin/metrics/signups")
    async def admin_signups(request: Request):
        """Daily signup counts for the last 90 days."""
        _require_admin(request)
        try:
            signups = await user_service.store.daily_signups(days=90)
            return {"signups": signups}
        except Exception as e:
            logger.error("Admin signups error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Signup data unavailable")

    @app.get("/admin/metrics/revenue")
    async def admin_revenue(request: Request):
        """MRR, ARR, and simple linear forecast."""
        _require_admin(request)
        try:
            from clsplusplus.tiers import Tier, TIER_PRICES
            tier_counts = await user_service.store.count_users_by_tier()

            mrr = sum(
                tier_counts.get(t.value, 0) * TIER_PRICES[t]
                for t in Tier
            )
            arr = mrr * 12

            # Simple forecast: assume current MRR for remaining FY months
            now = datetime.now()
            months_remaining = 12 - now.month
            fy_projected = arr  # Simplified: current run rate

            return {
                "mrr": round(mrr, 2),
                "arr": round(arr, 2),
                "fy_projected": round(fy_projected, 2),
                "months_remaining": months_remaining,
                "tier_counts": tier_counts,
            }
        except Exception as e:
            logger.error("Admin revenue error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Revenue data unavailable")

    @app.get("/admin/metrics/operations")
    async def admin_operations(request: Request):
        """Aggregate metering points across all users for current period."""
        _require_admin(request)
        try:
            aggregate = await _metrics.get_aggregate_metrics()
            from clsplusplus.cost_model import compute_cost
            total_cost = compute_cost(aggregate)

            return {
                "period": datetime.utcnow().strftime("%Y-%m"),
                "metrics": aggregate,
                "total_cost": round(total_cost, 4),
            }
        except Exception as e:
            logger.error("Admin operations error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Operations data unavailable")

    @app.get("/admin/metrics/users")
    async def admin_users(request: Request):
        """Per-user breakdown: tier, operations, cost, revenue."""
        _require_admin(request)
        try:
            from clsplusplus.tiers import Tier, TIER_PRICES
            from clsplusplus.cost_model import compute_cost

            users = await user_service.list_users(limit=500)
            result = []
            for u in users:
                user_metrics = await _metrics.get_user_metrics(u["id"])
                user_cost = compute_cost(user_metrics)
                user_revenue = TIER_PRICES.get(Tier(u["tier"]), 0)
                ops = sum(user_metrics.values())
                result.append({
                    "id": u["id"],
                    "email": u["email"],
                    "name": u["name"],
                    "tier": u["tier"],
                    "operations": ops,
                    "cost": round(user_cost, 4),
                    "revenue": round(user_revenue, 2),
                    "created_at": u["created_at"],
                })

            return {"users": result}
        except Exception as e:
            logger.error("Admin user list error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="User data unavailable")

    def _require_admin(request: Request):
        """Helper to enforce admin access on endpoints."""
        if not getattr(request.state, "is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")

    @app.patch("/admin/users/{user_id}")
    async def admin_update_user(request: Request, user_id: str = Path(...)):
        """Admin: update user fields (name, email, tier, is_admin)."""
        _require_admin(request)
        try:
            body = await request.json()
            fields = {}
            if "name" in body:
                fields["name"] = str(body["name"]).strip()
            if "email" in body:
                fields["email"] = str(body["email"]).strip().lower()
            if "tier" in body:
                if body["tier"] not in ("free", "pro", "business", "enterprise"):
                    raise HTTPException(status_code=400, detail="Invalid tier")
                fields["tier"] = body["tier"]
            if "is_admin" in body:
                fields["is_admin"] = bool(body["is_admin"])
            if not fields:
                raise HTTPException(status_code=400, detail="No fields to update")
            updated = await user_service.store.update_user(user_id, fields)
            if not updated:
                raise HTTPException(status_code=404, detail="User not found")
            return updated
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Admin user update error: %s", e)
            raise HTTPException(status_code=500, detail="User update failed")

    @app.delete("/admin/users/{user_id}")
    async def admin_delete_user(request: Request, user_id: str = Path(...)):
        """Admin: delete a user and all related data."""
        _require_admin(request)
        admin_id = getattr(request.state, "user_id", None)
        if user_id == admin_id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        try:
            import uuid as _uuid
            uid = _uuid.UUID(user_id)
            pool = await user_service.store.get_pool()
            async with pool.acquire() as conn:
                # Verify user exists
                exists = await conn.fetchval("SELECT 1 FROM users WHERE id = $1", uid)
                if not exists:
                    raise HTTPException(status_code=404, detail="User not found")
                # Dynamically find ALL tables with FK to users(id) and clean them
                fk_tables = await conn.fetch("""
                    SELECT DISTINCT tc.table_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.constraint_column_usage ccu
                      ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND ccu.table_name = 'users'
                      AND ccu.column_name = 'id'
                """)
                for row in fk_tables:
                    tbl = row["table_name"]
                    try:
                        await conn.execute(f"DELETE FROM {tbl} WHERE user_id = $1", uid)
                    except Exception:
                        pass
                # Also clean tables with user_id but no FK constraint
                for tbl in ("prompt_log", "context_log"):
                    try:
                        await conn.execute(f"DELETE FROM {tbl} WHERE user_id = $1", uid)
                    except Exception:
                        pass
                # Delete user
                result = await conn.execute("DELETE FROM users WHERE id = $1", uid)
                if result != "DELETE 1":
                    raise HTTPException(status_code=500, detail="Deletion failed")
            return {"detail": "User deleted successfully."}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Admin user delete error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail=f"User deletion failed: {type(e).__name__}: {e}")

    @app.get("/admin/metrics/user/{user_id}")
    async def admin_user_detail(request: Request, user_id: str = Path(...)):
        """Get detailed metrics for a specific user (admin only)."""
        _require_admin(request)
        try:
            user = await user_service.get_user(user_id)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user_metrics = await _metrics.get_user_metrics(user_id)
            from clsplusplus.cost_model import compute_cost
            from clsplusplus.tiers import Tier, TIER_PRICES
            user_cost = compute_cost(user_metrics)
            user_revenue = TIER_PRICES.get(Tier(user["tier"]), 0)
            return {
                "user": user,
                "metrics": user_metrics,
                "cost": round(user_cost, 4),
                "revenue": round(user_revenue, 2),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Admin user detail error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="User metrics unavailable")

    @app.get("/admin/metrics/extension")
    async def admin_extension(request: Request):
        """Browser extension analytics: installs, DAU/WAU/MAU, site usage."""
        _require_admin(request)
        try:
            return await _metrics.get_extension_analytics()
        except Exception as e:
            logger.error("Admin extension analytics error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Extension analytics unavailable")

    @app.get("/v1/stats/extension")
    async def public_extension_stats():
        """Public endpoint — returns extension install/active counts for social proof."""
        try:
            data = await _metrics.get_extension_analytics()
            return {"installs": data.get("installs_this_month", 0), "dau": data.get("dau", 0), "mau": data.get("mau", 0)}
        except Exception:
            return {"installs": 0, "dau": 0, "mau": 0}

    @app.get("/admin/metrics/storage")
    async def admin_storage(request: Request):
        """Storage metering: item counts across L0/L1/L2, namespaces."""
        _require_admin(request)
        try:
            l0_items = sum(len(items) for items in memory_service.engine._items.values())
            l0_namespaces = len(memory_service.engine._items)
            loaded_namespaces = len(memory_service._loaded_namespaces)

            # L1 count (if DB available)
            l1_count = 0
            l1_namespaces = 0
            try:
                ns_list = await memory_service.l1.list_namespaces()
                l1_namespaces = len(ns_list)
                for ns in ns_list[:50]:  # Cap to avoid slow query
                    l1_count += await memory_service.l1.count(ns)
            except Exception:
                pass

            return {
                "l0_items": l0_items,
                "l0_namespaces": l0_namespaces,
                "l1_items": l1_count,
                "l1_namespaces": l1_namespaces,
                "loaded_namespaces": loaded_namespaces,
            }
        except Exception as e:
            logger.error("Admin storage metrics error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Storage metrics unavailable")

    # =========================================================================
    # Admin — Launch waitlist view + lifecycle test runner
    # =========================================================================

    _WAITLIST_RESULTS_DIR = FilePath(__file__).resolve().parent.parent.parent / "website" / "tests" / "waitlist"
    _WAITLIST_RUNNER = FilePath(__file__).resolve().parent.parent.parent / "scripts" / "run_waitlist_tests.py"

    @app.get("/admin/waitlist")
    async def admin_waitlist_list(request: Request):
        """List waitlist visitors with status + computed display position."""
        _require_admin(request)
        try:
            rows = await waitlist_service.store.list_all(limit=500)
            offset = int(getattr(settings, "waitlist_queue_seed_offset", 0))
            # Compute display position per waiting row (offset-aware)
            waiting_sorted = sorted(
                [r for r in rows if r["status"] in ("waiting", "invited")],
                key=lambda r: r["created_at"],
            )
            pos_map = {r["id"]: i + 1 + offset for i, r in enumerate(waiting_sorted)}
            for r in rows:
                r["display_position"] = pos_map.get(r["id"])
            stats = await waitlist_service.stats()
            return {
                "visitors": rows,
                "stats": stats,
                "config": {
                    "max_active_users": int(getattr(settings, "max_active_users", 0)),
                    "seed_offset": offset,
                    "active_floor": int(getattr(settings, "waitlist_active_floor", 0)),
                    "dau_healthy_threshold": int(
                        getattr(settings, "waitlist_dau_healthy_threshold", 0)
                    ),
                    "promote_batch": int(getattr(settings, "waitlist_promote_batch", 1)),
                },
            }
        except Exception as e:
            logger.error("Admin waitlist list error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Waitlist data unavailable")

    @app.post("/admin/waitlist/promote")
    async def admin_waitlist_promote(request: Request):
        """Manual-promote the oldest N waiting visitors (admin override)."""
        _require_admin(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        batch = int((body or {}).get("batch", 1))
        try:
            invited = await waitlist_service.promote(
                batch=batch, base_url=settings.site_base_url
            )
            return {"invited": invited, "count": len(invited)}
        except Exception as e:
            logger.error("Admin waitlist promote error: %s: %s", type(e).__name__, e)
            raise HTTPException(status_code=500, detail="Promote failed")

    @app.post("/admin/tests/waitlist/run")
    async def admin_run_waitlist_tests(request: Request):
        """Subprocess the waitlist test runner, return the freshly-written result."""
        _require_admin(request)
        if not _WAITLIST_RUNNER.exists():
            raise HTTPException(status_code=500, detail="Test runner script missing")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                str(_WAITLIST_RUNNER),
                "--quiet",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(_WAITLIST_RUNNER.parent.parent),
            )
            stdout, stderr = await proc.communicate()
        except Exception as e:
            logger.error("Waitlist test runner spawn error: %s", e)
            raise HTTPException(status_code=500, detail="Failed to launch test runner")

        # Grab the latest result file (the one the runner just wrote)
        manifest_path = _WAITLIST_RESULTS_DIR / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"No manifest produced (exit={proc.returncode}): {stderr.decode()[:200]}",
            )
        import json as _json
        try:
            manifest = _json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
        if not manifest:
            raise HTTPException(status_code=500, detail="Manifest is empty")
        latest = manifest[0]
        result_file = _WAITLIST_RESULTS_DIR / latest["file"]
        payload = _json.loads(result_file.read_text()) if result_file.exists() else latest
        payload["stdout_tail"] = stdout.decode()[-400:]
        payload["stderr_tail"] = stderr.decode()[-400:]
        payload["exit_code"] = proc.returncode
        return payload

    @app.get("/admin/tests/waitlist/history")
    async def admin_waitlist_test_history(request: Request):
        """Rolling manifest of recent waitlist test runs (newest first)."""
        _require_admin(request)
        manifest_path = _WAITLIST_RESULTS_DIR / "manifest.json"
        if not manifest_path.exists():
            return {"runs": []}
        import json as _json
        try:
            return {"runs": _json.loads(manifest_path.read_text())}
        except Exception:
            return {"runs": []}

    @app.get("/admin/tests/waitlist/results/{run_id}")
    async def admin_waitlist_test_result(request: Request, run_id: str = Path(...)):
        """Full JSON for a single waitlist test run."""
        _require_admin(request)
        # Defensive: run_id must match our strftime format, no path traversal
        import re as _re
        if not _re.match(r"^[0-9]{8}T[0-9]{6}$", run_id):
            raise HTTPException(status_code=400, detail="Invalid run_id")
        result_file = _WAITLIST_RESULTS_DIR / f"{run_id}.json"
        if not result_file.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        import json as _json
        return _json.loads(result_file.read_text())

    # =========================================================================
    # RBAC Admin API — Roles, Groups, Users, Permissions
    # =========================================================================

    from clsplusplus.rbac_service import RBACService, ALL_SCOPES
    _rbac = RBACService(settings)

    @app.get("/admin/rbac/scopes")
    async def list_scopes(request: Request):
        _require_admin(request)
        return {"scopes": sorted(ALL_SCOPES)}

    @app.get("/admin/rbac/roles")
    async def list_roles(request: Request):
        _require_admin(request)
        return {"roles": await _rbac.store.list_roles()}

    @app.post("/admin/rbac/roles")
    async def create_role(request: Request):
        _require_admin(request)
        body = await request.json()
        role = await _rbac.store.create_role(body["name"], body.get("description", ""), body.get("scopes", []))
        return role

    @app.put("/admin/rbac/roles/{role_id}")
    async def update_role(request: Request, role_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        ok = await _rbac.store.update_role(role_id, body.get("description"), body.get("scopes"))
        if not ok:
            raise HTTPException(status_code=404, detail="Role not found or is a system role")
        return {"ok": True}

    @app.delete("/admin/rbac/roles/{role_id}")
    async def delete_role(request: Request, role_id: str = Path(...)):
        _require_admin(request)
        ok = await _rbac.store.delete_role(role_id)
        if not ok:
            raise HTTPException(status_code=400, detail="Cannot delete system role")
        return {"ok": True}

    @app.get("/admin/rbac/groups")
    async def list_groups(request: Request):
        _require_admin(request)
        return {"groups": await _rbac.store.list_groups()}

    @app.post("/admin/rbac/groups")
    async def create_group(request: Request):
        _require_admin(request)
        body = await request.json()
        group = await _rbac.store.create_group(body["name"], body.get("description", ""))
        return group

    @app.put("/admin/rbac/groups/{group_id}")
    async def update_group(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        ok = await _rbac.store.update_group(group_id, body.get("name"), body.get("description"))
        if not ok:
            raise HTTPException(status_code=404, detail="Group not found")
        return {"ok": True}

    @app.delete("/admin/rbac/groups/{group_id}")
    async def delete_group(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        ok = await _rbac.store.delete_group(group_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Group not found")
        return {"ok": True}

    @app.get("/admin/rbac/groups/{group_id}/roles")
    async def get_group_roles(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        return {"roles": await _rbac.store.get_group_roles(group_id)}

    @app.post("/admin/rbac/groups/{group_id}/roles")
    async def add_group_role(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        await _rbac.store.add_group_role(group_id, body["role_id"])
        await _rbac.invalidate_group_cache(group_id)
        return {"ok": True}

    @app.delete("/admin/rbac/groups/{group_id}/roles/{role_id}")
    async def remove_group_role(request: Request, group_id: str = Path(...), role_id: str = Path(...)):
        _require_admin(request)
        await _rbac.store.remove_group_role(group_id, role_id)
        await _rbac.invalidate_group_cache(group_id)
        return {"ok": True}

    @app.get("/admin/rbac/groups/{group_id}/members")
    async def get_group_members(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        return {"members": await _rbac.store.get_group_members(group_id)}

    @app.post("/admin/rbac/groups/{group_id}/members")
    async def add_group_member(request: Request, group_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        await _rbac.store.add_group_member(group_id, body["user_id"])
        await _rbac.invalidate_cache(body["user_id"])
        return {"ok": True}

    @app.delete("/admin/rbac/groups/{group_id}/members/{user_id}")
    async def remove_group_member(request: Request, group_id: str = Path(...), user_id: str = Path(...)):
        _require_admin(request)
        await _rbac.store.remove_group_member(group_id, user_id)
        await _rbac.invalidate_cache(user_id)
        return {"ok": True}

    @app.get("/admin/rbac/users/{user_id}/roles")
    async def get_user_roles(request: Request, user_id: str = Path(...)):
        _require_admin(request)
        return {"roles": await _rbac.store.get_user_roles(user_id)}

    @app.post("/admin/rbac/users/{user_id}/roles")
    async def add_user_role(request: Request, user_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        await _rbac.store.add_user_role(user_id, body["role_id"])
        await _rbac.invalidate_cache(user_id)
        return {"ok": True}

    @app.delete("/admin/rbac/users/{user_id}/roles/{role_id}")
    async def remove_user_role(request: Request, user_id: str = Path(...), role_id: str = Path(...)):
        _require_admin(request)
        await _rbac.store.remove_user_role(user_id, role_id)
        await _rbac.invalidate_cache(user_id)
        return {"ok": True}

    @app.get("/admin/rbac/users/{user_id}/permissions")
    async def get_user_permissions(request: Request, user_id: str = Path(...)):
        _require_admin(request)
        return {"permissions": await _rbac.store.get_user_permissions(user_id)}

    @app.post("/admin/rbac/users/{user_id}/permissions")
    async def set_user_permission(request: Request, user_id: str = Path(...)):
        _require_admin(request)
        body = await request.json()
        perm = await _rbac.store.set_user_permission(user_id, body["scope"], body.get("granted", True))
        await _rbac.invalidate_cache(user_id)
        return perm

    @app.delete("/admin/rbac/users/{user_id}/permissions/{permission_id}")
    async def remove_user_permission(request: Request, user_id: str = Path(...), permission_id: str = Path(...)):
        _require_admin(request)
        await _rbac.store.remove_user_permission(user_id, permission_id)
        await _rbac.invalidate_cache(user_id)
        return {"ok": True}

    @app.get("/admin/rbac/users/{user_id}/effective")
    async def get_effective_scopes(request: Request, user_id: str = Path(...)):
        _require_admin(request)
        scopes = await _rbac.get_effective_scopes(user_id)
        groups = await _rbac.store.get_user_groups(user_id)
        roles = await _rbac.store.get_user_roles(user_id)
        overrides = await _rbac.store.get_user_permissions(user_id)
        return {
            "scopes": sorted(scopes),
            "groups": groups,
            "direct_roles": roles,
            "overrides": overrides,
        }

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

    class _ChatSessionCreateReq(BaseModel):
        # namespace = the user's persistent brain.
        # All sessions for the same user must pass the same namespace so memory
        # flows across conversations and models.  Defaults to "default" so
        # direct API callers work without extra plumbing.
        namespace: str = "default"

    class _ChatMsgReq(BaseModel):
        message: str

    @app.post("/v1/chat/sessions")
    async def create_chat_session(req: _ChatSessionCreateReq):
        """Create a new chat session bound to the caller's persistent namespace.

        The namespace is the user's identity boundary — all sessions for the same
        user must use the same namespace so memory accumulates across conversations
        and across LLM models.
        """
        try:
            validate_namespace(req.namespace)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        sid = str(_uuid_mod.uuid4())
        _session_counter[0] += 1
        name = f"Chat {_session_counter[0]}"
        _sessions[sid] = _ChatSession(session_id=sid, name=name, namespace=req.namespace)
        return {"session_id": sid, "name": name, "namespace": req.namespace}

    @app.get("/v1/chat/sessions/{session_id}")
    async def get_chat_session(session_id: str = Path(..., min_length=1, max_length=64)):
        """Return a session with its full message history."""
        sess = _sessions.get(session_id)
        if not sess:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {
            "session_id": sess.session_id,
            "name": sess.name,
            "namespace": sess.namespace,
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

        await _record_usage("chat_message", request)

        return {
            "reply": reply,
            "memory_used": memory_used,
            "memory_count": memory_count,
            "debug": debug_info,
        }

    # =========================================================================
    # Trace / Call Graph API
    # =========================================================================

    @app.get("/v1/memory/traces")
    async def memory_traces(
        trace_id: Optional[str] = Query(default=None, description="If set, return full call graph for this id"),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        """Single endpoint: list recent traces, or one trace tree when trace_id is set."""
        if trace_id and trace_id.strip():
            trace = tracer.get(trace_id.strip())
            if not trace:
                raise HTTPException(
                    status_code=404,
                    detail=f"Trace {trace_id} not found (max {tracer.MAX_TRACES} traces kept in memory)",
                )
            return trace.to_dict()
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
        """Boot sequence:
        1. Preload every known namespace from L1 into PhaseMemoryEngine so the
           first request for any user is instant (no cold-start lag).
        2. Re-tune IVFFlat lists for the current row count (1M-ready).
        3. Start the periodic hippocampal replay loop (every 5 minutes).
        """
        # ── 0. Seed default admin user ─────────────────────────────────────
        try:
            await user_service.ensure_admin()
        except Exception as e:
            _api_logger.warning("Admin seeding failed (non-fatal): %s", e)

        # ── 1. Warm up all persisted namespaces from L1 ────────────────────
        await memory_service.startup_preload()

        # ── 2. Periodic hippocampal replay (consolidation / sleep cycle) ───
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

        # ── 3. Waitlist daily promoter ─────────────────────────────────────
        # Once per configured interval, promote the oldest waiting visitor(s)
        # iff DAU is below the healthy threshold, and sweep expired invites.
        async def _waitlist_promoter_loop():
            interval = int(getattr(settings, "waitlist_promote_interval_seconds", 86400))
            # Small startup delay so the pool is warm before the first tick.
            await asyncio.sleep(30)
            while True:
                try:
                    result = await waitlist_service.daily_promote_tick(
                        base_url=settings.site_base_url
                    )
                    if result.get("invited") or result.get("stale_invites_reset"):
                        _api_logger.info("Waitlist promote tick: %s", result)
                except Exception as e:
                    _api_logger.warning("Waitlist promote tick failed: %s", e)
                await asyncio.sleep(interval)

        asyncio.create_task(_waitlist_promoter_loop())

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
        # Close the shared httpx client used by LLM proxy routes
        try:
            http_client = getattr(_local_router, "_http_client", None)
            if http_client:
                await http_client.aclose()
        except Exception:
            pass

    # Register local/daemon routes (memory viewer, LLM proxies, WebSocket, installer)
    from clsplusplus.local_routes import create_local_router
    _local_router = create_local_router(memory_service, settings, metrics_emitter=_metrics)
    app.include_router(_local_router)

    # Serve website static files if the directory exists
    if _website_dir and FilePath(_website_dir).is_dir():
        app.mount("/", StaticFiles(directory=_website_dir, html=True), name="website")

    return app


app = create_app()

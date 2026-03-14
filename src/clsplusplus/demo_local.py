"""
Standalone demo API for local testing - NO Redis, NO Postgres.
Real Claude, OpenAI, Gemini only. Requires API keys in .env.
Run: uvicorn clsplusplus.demo_local:app --reload --port 8080
"""

import secrets
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clsplusplus.config import Settings
from clsplusplus.memory_phase import PhaseMemoryEngine

app = FastAPI(title="CLS++ Demo (Local)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# In-memory stores
# =============================================================================

# Phase Memory Engine — thermodynamic Gas → Liquid memory system
# F(θ, Σ, ρ, τ) = E_prediction − Σ·S_model + λ·L_landauer
_phase_settings = Settings()
_phase_engine = PhaseMemoryEngine(
    kT=_phase_settings.phase_kT,
    lambda_budget=_phase_settings.phase_lambda,
    tau_c1=_phase_settings.phase_tau_c1,
    tau_default=_phase_settings.phase_tau_default,
    tau_override=_phase_settings.phase_tau_override,
    strength_floor=_phase_settings.phase_strength_floor,
    capacity=_phase_settings.phase_capacity,
    beta_retrieval=_phase_settings.phase_beta_retrieval,
)

# Integration store: id -> integration dict
_integrations: dict[str, dict[str, Any]] = {}

# API keys: id -> key dict
_api_keys: dict[str, dict[str, Any]] = {}

# Webhooks: id -> webhook dict
_webhooks: dict[str, dict[str, Any]] = {}


# =============================================================================
# Demo Status
# =============================================================================

@app.get("/v1/demo/status")
async def status():
    s = Settings()
    return {
        "claude": bool(getattr(s, "anthropic_api_key", None)),
        "openai": bool(getattr(s, "openai_api_key", None)),
        "gemini": bool(getattr(s, "google_api_key", None)),
    }


# =============================================================================
# Integration Management (in-memory, no PostgreSQL)
# =============================================================================

class IntegrationCreateReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=1024)
    namespace: str = Field(default="default", min_length=1, max_length=64)
    owner_email: Optional[str] = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebhookCreateReq(BaseModel):
    url: str = Field(..., min_length=10, max_length=2048)
    events: list[str] = Field(default=["*"], max_length=50)
    description: str = Field(default="", max_length=1024)
    namespace_filter: Optional[str] = Field(default=None, max_length=64)


def _generate_api_key() -> str:
    return "cls_live_" + secrets.token_hex(16)


def _generate_webhook_secret() -> str:
    return "whsec_" + secrets.token_hex(20)


@app.post("/v1/integrations")
async def create_integration(req: IntegrationCreateReq):
    integration_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()

    integration = {
        "id": integration_id,
        "name": req.name,
        "description": req.description,
        "namespace": req.namespace,
        "owner_email": req.owner_email,
        "metadata": req.metadata,
        "status": "active",
        "created_at": now,
    }
    _integrations[integration_id] = integration

    # Auto-create first API key
    api_key = _generate_api_key()
    key_id = str(uuid4())
    key_record = {
        "id": key_id,
        "integration_id": integration_id,
        "key": api_key,
        "key_masked": api_key[:12] + "..." + api_key[-4:],
        "label": "default",
        "scopes": ["memory:read", "memory:write"],
        "created_at": now,
    }
    _api_keys[key_id] = key_record

    return {
        "integration": integration,
        "api_key": key_record,
        "_hint": "Save your API key now - it won't be shown again.",
    }


@app.get("/v1/integrations")
async def list_integrations(namespace: str = "default"):
    items = [i for i in _integrations.values() if i["namespace"] == namespace]
    return {"integrations": items}


@app.get("/v1/integrations/{integration_id}")
async def get_integration(integration_id: str = Path(...)):
    integration = _integrations.get(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    return integration


@app.delete("/v1/integrations/{integration_id}")
async def delete_integration(integration_id: str = Path(...)):
    if integration_id not in _integrations:
        raise HTTPException(status_code=404, detail="Integration not found")
    del _integrations[integration_id]
    return {"deleted": True, "integration_id": integration_id}


# --- Webhooks ---

@app.post("/v1/integrations/{integration_id}/webhooks")
async def create_webhook(req: WebhookCreateReq, integration_id: str = Path(...)):
    if integration_id not in _integrations:
        raise HTTPException(status_code=404, detail="Integration not found")

    webhook_id = str(uuid4())
    secret = _generate_webhook_secret()
    now = datetime.now(timezone.utc).isoformat()

    webhook = {
        "id": webhook_id,
        "integration_id": integration_id,
        "url": req.url,
        "events": req.events,
        "secret": secret,
        "description": req.description,
        "namespace_filter": req.namespace_filter,
        "status": "active",
        "created_at": now,
    }
    _webhooks[webhook_id] = webhook

    return {
        "webhook": webhook,
        "_hint": "Save your webhook signing secret now - it won't be shown again.",
    }


@app.get("/v1/integrations/{integration_id}/webhooks")
async def list_webhooks(integration_id: str = Path(...)):
    items = [w for w in _webhooks.values() if w["integration_id"] == integration_id]
    return {"webhooks": items}


@app.delete("/v1/integrations/{integration_id}/webhooks/{webhook_id}")
async def delete_webhook(integration_id: str = Path(...), webhook_id: str = Path(...)):
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    del _webhooks[webhook_id]
    return {"deleted": True, "webhook_id": webhook_id}


# =============================================================================
# Memory Cycle (uses Phase Engine exclusively)
# =============================================================================

class MemoryCycleReq(BaseModel):
    statements: list[str] = Field(..., min_length=1, max_length=20)
    queries: list[str] = Field(..., min_length=1, max_length=10)
    models: list[str] = Field(default=["claude", "openai"], max_length=3)
    namespace: str = Field(default="cycle-test", min_length=1, max_length=64)


@app.post("/v1/demo/memory-cycle")
async def memory_cycle(req: MemoryCycleReq):
    for m in req.models:
        if m not in ("claude", "openai", "gemini"):
            raise HTTPException(status_code=400, detail=f"Invalid model: {m}")

    settings = Settings()
    cycle_id = str(uuid4())
    extraction_caller = _make_extraction_caller(settings)

    # Phase 1: ENCODE — ingest through the thermodynamic attention gate
    encode_items = []
    for stmt in req.statements:
        item = await _phase_engine.ingest(stmt, req.namespace, extraction_caller)
        if item:
            encode_items.append({
                "id": item.id,
                "text": item.fact.raw_text,
                "strength": round(item.consolidation_strength, 4),
                "phase": "liquid" if item.consolidation_strength >= _phase_engine.STRENGTH_FLOOR else "gas",
            })

    # Phase 2: RETRIEVE — free-energy-ranked search
    retrieve_results = []
    for query in req.queries:
        results = _phase_engine.search(query, req.namespace, limit=10)
        retrieve_results.append({
            "query": query,
            "found": len(results),
            "items": [{"id": item.id, "text": item.fact.raw_text,
                        "strength": round(item.consolidation_strength, 4),
                        "score": round(score, 4)}
                       for score, item in results[:5]],
        })

    total_found = sum(r["found"] for r in retrieve_results)

    # Phase 3: AUGMENT (real LLM calls with phase-engine context)
    augment_results = {}
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini
    callers = {"claude": call_claude, "openai": call_openai, "gemini": call_gemini}

    for model in req.models:
        model_results = []
        for query in req.queries[:2]:
            memory_context, debug_items = _phase_engine.build_augmented_context(
                query, req.namespace, limit=5
            )
            system_prompt = f"You are a helpful assistant.\n\n{memory_context}\n\nAnswer naturally."
            try:
                reply = await callers[model](settings, system_prompt, query)
                model_results.append({
                    "query": query, "response": reply,
                    "memory_context_items": len(debug_items), "memory_used": len(debug_items) > 0,
                })
            except Exception as e:
                model_results.append({"query": query, "error": str(e), "memory_used": False})
        augment_results[model] = model_results

    # Verdict
    encode_ok = len(encode_items) > 0
    retrieve_ok = total_found > 0
    augment_ok = any(
        any(r.get("memory_used", False) for r in results)
        for results in augment_results.values()
    )

    verdict = "PASS" if (encode_ok and retrieve_ok and augment_ok) else (
        "PARTIAL" if (encode_ok and retrieve_ok) else "FAIL"
    )

    return {
        "cycle_id": cycle_id,
        "namespace": req.namespace,
        "models": req.models,
        "phases": {
            "encode": {"stored": len(encode_items), "total": len(req.statements), "items": encode_items},
            "retrieve": {"queries": len(req.queries), "total_found": total_found, "results": retrieve_results},
            "augment": augment_results,
        },
        "verdict": verdict,
        "phase_debug": _phase_engine.get_phase_debug(req.namespace),
    }


# =============================================================================
# Chat Sessions + LLM Routing Layer
# =============================================================================

# Session store: session_id -> { id, name, namespace, created_at, messages }
_sessions: dict[str, dict[str, Any]] = {}

# LLM priority for automatic routing with failover
_LLM_PRIORITY = ["openai", "claude", "gemini"]

_ERROR_MARKERS = [
    "An error occurred",
    "Add CLS_ANTHROPIC_API_KEY",
    "Add CLS_OPENAI_API_KEY",
    "Add CLS_GOOGLE_API_KEY",
    "No response",
    "content may have been blocked",
    "credit balance is too low",
]


def _is_error_reply(reply: str) -> bool:
    return any(marker in reply for marker in _ERROR_MARKERS)


async def _route_to_llm(settings: Settings, system: str, user_msg: str) -> tuple[str, str]:
    """Try each LLM in priority order. First success wins. User never knows.
    Returns (reply, model_used)."""
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini

    callers = {"claude": call_claude, "openai": call_openai, "gemini": call_gemini}
    last_reply = ""

    for model_name in _LLM_PRIORITY:
        try:
            reply = await callers[model_name](settings, system, user_msg)
            if not _is_error_reply(reply):
                return reply, model_name
            last_reply = reply
        except Exception:
            continue

    # All failed
    if last_reply:
        return last_reply, "unknown"
    return "I'm having trouble connecting right now. Please try again in a moment.", "none"


def _make_extraction_caller(settings: Settings):
    """Create an LLM caller for the phase engine's attention gate.

    Returns an async function(system, user_msg) → str that routes
    through the same failover pipeline as chat responses.
    """
    async def caller(system: str, user_msg: str) -> str:
        reply, _ = await _route_to_llm(settings, system, user_msg)
        return reply
    return caller


class SessionMessageReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


# Global memory namespace — shared across ALL sessions for the same user.
# Conversation history stays per-session; memory is cross-session.
_GLOBAL_NS = "global"


@app.post("/v1/chat/sessions")
async def create_session():
    session_id = str(uuid4())
    short_id = session_id[:4]
    now = datetime.now(timezone.utc).isoformat()

    session = {
        "id": session_id,
        "name": f"Chat {short_id}",
        "namespace": _GLOBAL_NS,
        "created_at": now,
        "messages": [],
    }
    _sessions[session_id] = session
    return {"session_id": session_id, "name": session["name"], "namespace": _GLOBAL_NS, "created_at": now}


@app.get("/v1/chat/sessions")
async def list_sessions():
    sessions = sorted(_sessions.values(), key=lambda s: s["created_at"], reverse=True)
    return {
        "sessions": [
            {"session_id": s["id"], "name": s["name"], "namespace": s["namespace"],
             "created_at": s["created_at"], "message_count": len(s["messages"])}
            for s in sessions
        ]
    }


@app.get("/v1/chat/sessions/{session_id}")
async def get_session(session_id: str = Path(...)):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session["id"],
        "name": session["name"],
        "namespace": session["namespace"],
        "created_at": session["created_at"],
        "messages": session["messages"],
    }


@app.delete("/v1/chat/sessions/{session_id}")
async def delete_session(session_id: str = Path(...)):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    # Don't wipe global memory when deleting a session — memory persists across sessions
    del _sessions[session_id]
    return {"deleted": True, "session_id": session_id}


@app.post("/v1/chat/sessions/{session_id}/message")
async def send_message(req: SessionMessageReq, session_id: str = Path(...)):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = req.message.strip()
    now = datetime.now(timezone.utc).isoformat()
    settings = Settings()

    # 1. Store user message in session conversation history
    session["messages"].append({"role": "user", "content": user_msg, "timestamp": now})

    # 2. PHASE ENGINE: Ingest message through the Gas → Liquid attention gate.
    #    The phase engine uses the LLM to extract structured facts (Fact dataclass).
    #    If the message is a factual statement, it condenses from gas to liquid.
    #    If it's a question/greeting, it stays gas (returns None, not stored).
    #    Surprise (Σ) is computed as KL divergence against existing beliefs.
    #    Contradicted memories receive irreversible surprise damage.
    extraction_caller = _make_extraction_caller(settings)
    await _phase_engine.ingest(user_msg, _GLOBAL_NS, extraction_caller)  # returns list now

    # 3. PHASE ENGINE: Retrieve via free energy ranking.
    #    Score = -F(item) × relevance(query, item)
    #    Items below strength_floor (gas phase) are excluded.
    #    No "NEWEST FIRST" hack — the physics handles conflict resolution.
    #    NOTE: Only call build_augmented_context (which calls search() internally).
    #    Do NOT call search() separately — that would double-count retrieval.
    memory_context, phase_debug_items = _phase_engine.build_augmented_context(
        user_msg, _GLOBAL_NS, limit=5
    )
    phase_results = phase_debug_items  # Already computed by build_augmented_context

    # 4. Build conversation history for THIS session only (last 20 messages)
    recent_messages = session["messages"][-20:]
    convo_lines = []
    for m in recent_messages[:-1]:  # Exclude the message we just added
        role_label = "User" if m["role"] == "user" else "Assistant"
        convo_lines.append(f"{role_label}: {m['content']}")
    conversation_history = "\n".join(convo_lines) if convo_lines else ""

    # 5. Build augmented system prompt — physics-driven, no hacks
    system_parts = [
        "You are a helpful, friendly assistant. Chat naturally and conversationally.",
    ]
    if conversation_history:
        system_parts.append(f"Recent conversation in this chat:\n{conversation_history}")
    if phase_results:
        system_parts.append(memory_context)
    system_parts.append(
        "Use the strongest-recalled memories as ground truth. "
        "Respond naturally. Never mention 'memory' or 'context' explicitly."
    )
    system = "\n\n".join(system_parts)

    # 6. Route to LLM with automatic failover
    reply, model_used = await _route_to_llm(settings, system, user_msg)

    # 7. Store AI response in session history (NOT in phase memory — only user facts)
    reply_ts = datetime.now(timezone.utc).isoformat()
    memory_used = len(phase_results) > 0
    session["messages"].append({
        "role": "assistant", "content": reply, "timestamp": reply_ts,
        "memory_used": memory_used, "memory_count": len(phase_results),
    })

    # 8. Return response + full thermodynamic debug info
    phase_debug = _phase_engine.get_phase_debug(_GLOBAL_NS)

    return {
        "reply": reply,
        "memory_used": memory_used,
        "memory_count": len(phase_results),
        "debug": {
            "model_used": model_used,
            "augmented_prompt": system,
            "user_message": user_msg,
            "memory_searched": [d["text"] for d in phase_debug_items],
            "conversation_history_lines": len(convo_lines),
            "memory_store": phase_debug.get("items", []),
            "phase_dynamics": phase_debug,
        },
    }


# =============================================================================
# Health
# =============================================================================

@app.get("/v1/memory/health")
async def health():
    return {
        "status": "healthy",
        "stores": {
            "L0": {"status": "healthy", "store": "in-memory"},
            "L1": {"status": "healthy", "store": "in-memory"},
            "L2": {"status": "healthy", "store": "in-memory"},
            "L3": {"status": "healthy", "store": "in-memory"},
        },
        "mode": "demo-local",
    }


@app.get("/")
async def root():
    return {
        "name": "CLS++ Demo API (Local)",
        "version": "0.1.0",
        "mode": "demo-local (in-memory, no Redis/Postgres)",
        "docs": "/docs",
        "health": "/v1/memory/health",
    }

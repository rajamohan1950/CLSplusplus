"""
Standalone demo API for local testing - NO Redis, NO Postgres.
Real Claude, OpenAI, Gemini only. Requires API keys in .env.
Run: uvicorn clsplusplus.demo_local:app --reload --port 8080
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clsplusplus.config import Settings

app = FastAPI(title="CLS++ Demo (Local)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# In-memory stores (replaces Redis + PostgreSQL for local testing)
# =============================================================================

# Memory store: namespace -> list of {id, text, confidence, ...}
_memory: dict[str, list[dict[str, Any]]] = {}

# Integration store: id -> integration dict
_integrations: dict[str, dict[str, Any]] = {}

# API keys: id -> key dict
_api_keys: dict[str, dict[str, Any]] = {}

# Webhooks: id -> webhook dict
_webhooks: dict[str, dict[str, Any]] = {}


# =============================================================================
# Chat (existing)
# =============================================================================

class ChatRequest(BaseModel):
    model: str
    message: str
    namespace: str = "demo"


def _get_memory_context(namespace: str, query: str) -> str:
    items = _memory.get(namespace, [])
    return "\n".join(f"- {t['text']}" for t in items) if items else "No prior context yet."


def _store_memory(namespace: str, text: str, source: str = "user",
                  salience: float = 0.5, authority: float = 0.5) -> dict:
    item = {
        "id": str(uuid4()),
        "text": text,
        "namespace": namespace,
        "source": source,
        "store_level": "L0",
        "confidence": min(salience * 0.6 + authority * 0.4, 1.0),
        "salience": salience,
        "authority": authority,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _memory.setdefault(namespace, []).append(item)
    return item


def _store_if_statement(namespace: str, message: str) -> None:
    is_q = "?" in message or any(
        message.strip().lower().startswith(w)
        for w in ("what", "who", "where", "when", "how", "which", "is my", "do you")
    )
    if not is_q:
        _store_memory(namespace, message)


def _search_memory(namespace: str, query: str, limit: int = 10) -> list[dict]:
    items = _memory.get(namespace, [])
    # Simple keyword matching with RECENCY BOOST (no embeddings in demo mode)
    # Newer items with same relevance score win — latest fact overrides older ones.
    query_words = set(query.lower().split())
    scored = []
    for idx, item in enumerate(items):
        text_words = set(item["text"].lower().split())
        overlap = len(query_words & text_words)
        relevance = overlap / max(len(query_words), 1)
        if relevance > 0:
            # Recency boost: later items (higher idx) get a small bonus
            recency = idx / max(len(items), 1) * 0.1
            scored.append((relevance + recency, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


@app.get("/v1/demo/status")
async def status():
    s = Settings()
    return {
        "claude": bool(getattr(s, "anthropic_api_key", None)),
        "openai": bool(getattr(s, "openai_api_key", None)),
        "gemini": bool(getattr(s, "google_api_key", None)),
    }


@app.post("/v1/demo/chat")
async def chat(req: ChatRequest):
    if req.model not in ("claude", "openai", "gemini"):
        return {"error": "model must be claude, openai, or gemini"}
    if not req.message.strip():
        return {"error": "message required"}

    settings = Settings()
    _store_if_statement(req.namespace, req.message.strip())
    memory_context = _get_memory_context(req.namespace, req.message)
    system = f"""You are a friendly, helpful assistant. Chat naturally.
When the user tells you something, reply naturally. When they ask a question, use this context if relevant:
{memory_context}
Respond naturally. Don't mention "memory" or "context"."""

    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini
    if req.model == "claude":
        reply = await call_claude(settings, system, req.message.strip())
    elif req.model == "openai":
        reply = await call_openai(settings, system, req.message.strip())
    else:
        reply = await call_gemini(settings, system, req.message.strip())

    return {"model": req.model, "reply": reply}


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
# Memory Cycle (in-memory, real LLM calls)
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
    phases = {}

    # Phase 1: ENCODE
    encode_items = []
    for stmt in req.statements:
        item = _store_memory(req.namespace, stmt, source="memory-cycle",
                             salience=0.9, authority=0.8)
        encode_items.append({
            "id": item["id"],
            "text": item["text"],
            "store_level": item["store_level"],
            "confidence": item["confidence"],
        })

    phases["encode"] = {
        "stored": len(encode_items),
        "total": len(req.statements),
        "items": encode_items,
    }

    # Phase 2: RETRIEVE
    retrieve_results = []
    for query in req.queries:
        found = _search_memory(req.namespace, query, limit=10)
        retrieve_results.append({
            "query": query,
            "found": len(found),
            "items": [{"id": i["id"], "text": i["text"], "confidence": i["confidence"]}
                      for i in found[:5]],
        })

    total_found = sum(r["found"] for r in retrieve_results)
    all_items = [item for r in retrieve_results for item in r.get("items", [])]
    avg_confidence = (sum(i["confidence"] for i in all_items) / len(all_items)) if all_items else 0.0

    phases["retrieve"] = {
        "queries": len(req.queries),
        "total_found": total_found,
        "confidence_avg": round(avg_confidence, 3),
        "results": retrieve_results,
    }

    # Phase 3: AUGMENT (real LLM calls)
    augment_results = {}
    from clsplusplus.demo_llm_calls import call_claude, call_openai, call_gemini

    for model in req.models:
        model_results = []
        for query in req.queries[:2]:
            found = _search_memory(req.namespace, query, limit=8)
            memory_context = "\n".join(f"- {i['text']}" for i in found) if found else "No prior context."

            system_prompt = f"""You are a helpful assistant. Use this context to answer:
{memory_context}
Answer naturally based on the context provided."""

            try:
                if model == "claude":
                    reply = await call_claude(settings, system_prompt, query)
                elif model == "openai":
                    reply = await call_openai(settings, system_prompt, query)
                elif model == "gemini":
                    reply = await call_gemini(settings, system_prompt, query)
                else:
                    reply = f"Unknown model: {model}"

                model_results.append({
                    "query": query,
                    "response": reply,
                    "memory_context_items": len(found),
                    "memory_used": len(found) > 0,
                })
            except Exception as e:
                model_results.append({
                    "query": query,
                    "error": str(e),
                    "memory_used": False,
                })

        augment_results[model] = model_results

    phases["augment"] = augment_results

    # Phase 4: CROSS-SESSION PERSISTENCE
    cross_session_results = []
    for query in req.queries[:2]:
        found = _search_memory(req.namespace, query, limit=10)
        cross_session_results.append({
            "query": query,
            "found": len(found),
            "persisted": len(found) > 0,
        })

    all_persisted = all(r.get("persisted", False) for r in cross_session_results)
    total_cross = sum(r["found"] for r in cross_session_results)

    phases["cross_session"] = {
        "namespace": req.namespace,
        "memories_persisted": all_persisted,
        "items_found": total_cross,
        "results": cross_session_results,
    }

    # Verdict
    encode_ok = phases["encode"]["stored"] == phases["encode"]["total"]
    retrieve_ok = phases["retrieve"]["total_found"] > 0
    augment_ok = any(
        any(r.get("memory_used", False) for r in results)
        for results in phases["augment"].values()
    )
    persist_ok = phases["cross_session"]["memories_persisted"]

    if encode_ok and retrieve_ok and augment_ok and persist_ok:
        verdict = "PASS"
    elif encode_ok and retrieve_ok:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "cycle_id": cycle_id,
        "namespace": req.namespace,
        "models": req.models,
        "phases": phases,
        "verdict": verdict,
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

    # 1. Store user message in history; store in GLOBAL memory (only statements, not questions)
    session["messages"].append({"role": "user", "content": user_msg, "timestamp": now})
    _store_if_statement(_GLOBAL_NS, user_msg)

    # 2. Search GLOBAL memory for relevant context (already sorted: newest-relevant first)
    found = _search_memory(_GLOBAL_NS, user_msg, limit=10)
    memory_snippets = [i["text"] for i in found]
    memory_context = "\n".join(f"- {t}" for t in memory_snippets) if found else "No prior context yet."

    # 3. Build conversation history for THIS session only (last 20 messages)
    recent_messages = session["messages"][-20:]
    convo_lines = []
    for m in recent_messages[:-1]:  # Exclude the message we just added
        role_label = "User" if m["role"] == "user" else "Assistant"
        convo_lines.append(f"{role_label}: {m['content']}")
    conversation_history = "\n".join(convo_lines) if convo_lines else ""

    # 4. Build augmented system prompt with BOTH memory + conversation history
    system_parts = [
        "You are a helpful, friendly assistant. Chat naturally and conversationally.",
    ]
    if conversation_history:
        system_parts.append(f"Recent conversation in this chat:\n{conversation_history}")
    if found:
        system_parts.append(f"Memory (facts the user told you, listed NEWEST FIRST):\n{memory_context}")
    system_parts.append("""CRITICAL RULE: When memory contains conflicting facts about the same topic, ONLY use the NEWEST fact (listed first). The newest fact COMPLETELY REPLACES any older conflicting facts. For example, if memory says "- X eats banana only" then "- X eats apple", the answer is ONLY banana because that fact is newer and the word "only" means exclusively that.
Respond naturally. Never mention 'memory' or 'context' explicitly.""")
    system = "\n\n".join(system_parts)

    # 5. Route to LLM with automatic failover
    settings = Settings()
    reply, model_used = await _route_to_llm(settings, system, user_msg)

    # 6. Store AI response in history + memory
    reply_ts = datetime.now(timezone.utc).isoformat()
    session["messages"].append({
        "role": "assistant", "content": reply, "timestamp": reply_ts,
        "memory_used": len(found) > 0, "memory_count": len(found),
    })
    # Don't store assistant replies in memory — only user statements are facts.
    # Assistant replies are noise ("That's great! Apples are healthy...") and pollute search.

    # 7. Return response + debug info
    all_memory = _memory.get(_GLOBAL_NS, [])

    return {
        "reply": reply,
        "memory_used": len(found) > 0,
        "memory_count": len(found),
        "debug": {
            "model_used": model_used,
            "augmented_prompt": system,
            "user_message": user_msg,
            "memory_searched": memory_snippets,
            "conversation_history_lines": len(convo_lines),
            "memory_store": [
                {"text": item["text"], "source": item["source"], "created_at": item["created_at"]}
                for item in all_memory
            ],
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

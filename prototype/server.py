"""
CLS++ Memory Proxy Server — with Model Tracking + Classification
"""
import sys, os, json, re, hashlib
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from clsplusplus.memory_phase import PhaseMemoryEngine

engine = PhaseMemoryEngine()

# ── Extended memory store (wraps engine items with metadata) ──────────────
# { uid: [ {id, text, category, source_model, ts, strength} ] }
memory_log: dict[str, list[dict]] = defaultdict(list)

# ── Context bridge log ─────────────────────────────────────────────────────
# { uid: [ {model, memories_sent:[text], query, ts} ] }
context_log: dict[str, list[dict]] = defaultdict(list)

# ── WebSocket clients ──────────────────────────────────────────────────────
_ws_clients: dict[str, list[WebSocket]] = defaultdict(list)

OPENAI_API_KEY    = os.getenv("CLS_OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("CLS_ANTHROPIC_API_KEY", "")
OPENAI_BASE       = "https://api.openai.com"
ANTHROPIC_BASE    = "https://api.anthropic.com"

app = FastAPI(title="CLS++ Memory Proxy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Mock LLM endpoints for Chrome extension E2E (extension/e2e/) ───────────
def _extract_chatgpt_last_part(body: dict) -> str:
    msgs = body.get("messages") or []
    if not msgs:
        return ""
    last = msgs[-1]
    parts = (last.get("content") or {}).get("parts")
    if parts and isinstance(parts[0], str):
        return parts[0]
    return ""


@app.post("/backend-api/conversation")
@app.post("/backend-api/f/conversation")
async def mock_chatgpt_conversation(request: Request):
    """Echo ChatGPT-shaped body so Playwright can assert CLS++ injected context."""
    raw = await request.body()
    try:
        b = json.loads(raw)
    except Exception:
        b = {}
    last_text = _extract_chatgpt_last_part(b)
    return {
        "mock": "chatgpt",
        "last_user_message_text": last_text,
        "injection_ok": (
            "BLUEBADGER" in last_text
            or "bluebadger" in last_text.lower()
        )
        and len(last_text) > len("What is my secret codename?") + 20,
    }


@app.post("/api/e2e/chat_conversations/{conv_id}/completion")
async def mock_claude_completion(request: Request, conv_id: str):
    """Echo Claude-shaped prompt so E2E can assert memory injection."""
    try:
        b = await request.json()
    except Exception:
        b = {}
    prompt = b.get("prompt") or ""
    return {
        "mock": "claude",
        "prompt_echo": prompt,
        "injection_ok": "Springfield" in prompt and len(prompt) > 40,
    }


app.mount("/ui", StaticFiles(directory=os.path.dirname(__file__), html=True), name="ui")


# ── Classification ─────────────────────────────────────────────────────────
CATEGORIES = {
    "Identity":    r"\b(name|called|i am|i'm|born|from|live|based|nationality|age|old)\b",
    "Preference":  r"\b(prefer|like|love|hate|enjoy|dislike|favourite|favorite|don't like|want)\b",
    "Work":        r"\b(work|job|role|company|employer|position|career|profession|office|salary)\b",
    "Project":     r"\b(build|building|startup|product|project|launch|app|platform|startup|mvp)\b",
    "Relationship":r"\b(wife|husband|partner|friend|colleague|team|family|brother|sister|son|daughter|manager|boss)\b",
    "Goal":        r"\b(goal|plan|want to|trying|aim|vision|mission|target|milestone)\b",
    "Temporal":    r"\b(yesterday|today|tomorrow|last week|next week|deadline|schedule|meeting|remind)\b",
    "Context":     r"\b(remember|note|context|background|mentioned|told|said|asked|discussed)\b",
}

def classify(text: str) -> str:
    t = text.lower()
    for cat, pattern in CATEGORIES.items():
        if re.search(pattern, t):
            return cat
    return "General"

def _ns(request: Request, body: dict) -> str:
    return (request.headers.get("X-User-Id")
            or body.get("user")
            or hashlib.sha256(request.client.host.encode()).hexdigest()[:16])

def _user_text(messages: list) -> str:
    parts = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str): parts.append(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b["text"])
    return " ".join(parts).strip()

def _inject(messages: list, memories: list) -> list:
    if not memories: return messages
    block = "[CLS++ Memory]\n" + "\n".join(f"• {i.fact.raw_text}" for _, i in memories)
    msgs = list(messages)
    if msgs and msgs[0].get("role") == "system":
        msgs[0] = {**msgs[0], "content": block + "\n\n" + msgs[0]["content"]}
    else:
        msgs.insert(0, {"role": "system", "content": block})
    return msgs

def _assistant_facts(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kw = ["you ", "your ", "you've", "you are", "you were", "you like",
          "you prefer", "you work", "you use", "you have", "you're"]
    return [s.strip() for s in sentences
            if any(k in s.lower() for k in kw) and len(s.strip()) > 20][:3]

def _item_phase(item) -> str:
    """Determine thermodynamic phase of a PhaseMemoryItem."""
    STRENGTH_FLOOR = 0.05
    if item.consolidation_strength < STRENGTH_FLOOR:
        return "gas"
    if item.schema_meta is not None:
        # Check glass (converged entropy)
        if hasattr(item.schema_meta, 'H_history') and len(item.schema_meta.H_history) >= 3:
            vals = item.schema_meta.H_history[-3:]
            mean = sum(vals) / len(vals)
            if mean > 0 and max(abs(v - mean) for v in vals) / mean < 0.01:
                return "glass"
        return "solid"
    return "liquid"

def _phase_layer(phase: str) -> str:
    return {"gas": "L0", "liquid": "L1", "solid": "L2", "glass": "L3"}.get(phase, "L1")

async def _store(uid: str, text: str, model: str, source: str = "user"):
    if len(text.strip()) < 6: return None
    item = engine.store(text, uid)
    if item is None: return None
    cat = classify(text)
    phase = _item_phase(item)
    entry = {
        "id": item.id,
        "text": item.fact.raw_text,
        "subject": item.fact.subject,
        "relation": item.fact.relation,
        "value": item.fact.value,
        "category": cat,
        "source_model": model,
        "source": source,
        "strength": round(item.consolidation_strength, 3),
        "phase": phase,
        "layer": _phase_layer(phase),
        "tau": round(getattr(item, 'tau', 0), 2),
        "retrieval_count": getattr(item, 'retrieval_count', 0),
        "ts": datetime.utcnow().isoformat(),
    }
    memory_log[uid].append(entry)

    payload = json.dumps({"type": "memory", **entry})
    dead = []
    for ws in _ws_clients.get(uid, []):
        try:   await ws.send_text(payload)
        except: dead.append(ws)
    for ws in dead: _ws_clients[uid].remove(ws)
    return entry

def _log_context(uid: str, model: str, query: str, memories: list):
    context_log[uid].append({
        "model": model,
        "query": query[:120],
        "memories_sent": [i.fact.raw_text for _, i in memories],
        "count": len(memories),
        "ts": datetime.utcnow().isoformat(),
    })


# ── OpenAI proxy ───────────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def openai_chat(request: Request):
    body  = await request.json()
    uid   = _ns(request, body)
    model_name = body.get("model", "openai")
    query = _user_text(body.get("messages", []))
    mems  = engine.search(query, uid, limit=6) if query else []

    _log_context(uid, model_name, query, mems)
    body["messages"] = _inject(body.get("messages", []), mems)

    auth  = request.headers.get("Authorization") or f"Bearer {OPENAI_API_KEY}"
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions",
            json=body, headers={"Authorization": auth, "Content-Type": "application/json"})

    if query: await _store(uid, query, model_name, "user")
    try:
        content = resp.json()["choices"][0]["message"]["content"]
        for f in _assistant_facts(content):
            await _store(uid, f, model_name, "assistant")
    except: pass
    return Response(content=resp.content, media_type="application/json", status_code=resp.status_code)


# ── Anthropic proxy ────────────────────────────────────────────────────────
@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    body  = await request.json()
    uid   = _ns(request, body)
    model_name = body.get("model", "claude")
    query = _user_text(body.get("messages", []))
    mems  = engine.search(query, uid, limit=6) if query else []

    _log_context(uid, model_name, query, mems)
    if mems:
        mem_block = "[CLS++ Memory]\n" + "\n".join(f"• {i.fact.raw_text}" for _, i in mems)
        body["system"] = (mem_block + "\n\n" + body.get("system", "")).strip()

    auth_key = (request.headers.get("x-api-key")
                or request.headers.get("Authorization","").replace("Bearer ","")
                or ANTHROPIC_API_KEY)
    async with httpx.AsyncClient(timeout=90) as client:
        resp = await client.post(f"{ANTHROPIC_BASE}/v1/messages", json=body,
            headers={"x-api-key": auth_key, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"})

    if query: await _store(uid, query, model_name, "user")
    try:
        content = resp.json()["content"][0]["text"]
        for f in _assistant_facts(content):
            await _store(uid, f, model_name, "assistant")
    except: pass
    return Response(content=resp.content, media_type="application/json", status_code=resp.status_code)


# ── Demo chat (powers memory.html live chat) ────────────────────────────────
@app.post("/api/chat/{uid}")
async def demo_chat(uid: str, request: Request):
    body    = await request.json()
    message = body.get("message","").strip()
    model   = body.get("model", "claude-haiku-4-5")
    if not message: return JSONResponse({"error":"message required"}, 400)

    mems = engine.search(message, uid, limit=6)
    _log_context(uid, model, message, mems)
    await _store(uid, message, model, "user")

    system = "You are a helpful AI assistant with memory about this user. Be concise."
    if mems:
        system = "[What you remember about this user]\n" + \
                 "\n".join(f"• {i.fact.raw_text}" for _, i in mems) + "\n\n" + system

    reply = "Error reaching AI."
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
                # ── OpenAI route ──────────────────────────────────────────
                resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions",
                    json={"model": model, "max_tokens": 512,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user",   "content": message}]},
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                             "Content-Type": "application/json"})
                reply = resp.json()["choices"][0]["message"]["content"]
            else:
                # ── Anthropic route ───────────────────────────────────────
                resp = await client.post(f"{ANTHROPIC_BASE}/v1/messages",
                    json={"model": model, "max_tokens": 512, "system": system,
                          "messages": [{"role": "user", "content": message}]},
                    headers={"x-api-key": ANTHROPIC_API_KEY,
                             "anthropic-version": "2023-06-01",
                             "Content-Type": "application/json"})
                reply = resp.json()["content"][0]["text"]
    except Exception as e:
        reply = f"Error reaching AI: {e}"

    for f in _assistant_facts(reply):
        await _store(uid, f, model, "assistant")

    return {
        "reply": reply,
        "memories_used": len(mems),
        "memory_snippets": [i.fact.raw_text for _, i in mems],
        "model": model,
    }


# ── REST: memories with full metadata ─────────────────────────────────────

@app.get("/api/memories/{uid}")
async def get_memories(uid: str, model: str = "", category: str = ""):
    """Return all memories for the UI viewer.
    Trusts the engine for phase/strength — no external fact filters needed.
    The engine already handles contradiction suppression and phase decay.
    """
    items = engine._items.get(uid, [])
    log_map = {e["id"]: e for e in memory_log.get(uid, [])}

    entries = []
    for item in items:
        # Skip gas-phase items (decayed/forgotten)
        phase = _item_phase(item)
        if phase == "gas":
            continue
        log_entry = log_map.get(item.id, {})
        entry = {
            "id": item.id,
            "text": item.fact.raw_text,
            "subject": item.fact.subject,
            "relation": item.fact.relation,
            "value": item.fact.value,
            "category": log_entry.get("category", classify(item.fact.raw_text)),
            "source_model": log_entry.get("source_model", "unknown"),
            "source": log_entry.get("source", "user"),
            "strength": round(item.consolidation_strength, 3),
            "phase": phase,
            "layer": _phase_layer(phase),
            "tau": round(getattr(item, 'tau', 0), 2),
            "retrieval_count": getattr(item, 'retrieval_count', 0),
            "ts": log_entry.get("ts", ""),
        }
        entries.append(entry)

    if model:    entries = [e for e in entries if e["source_model"] == model]
    if category: entries = [e for e in entries if e["category"] == category]
    models = list({e["source_model"] for e in entries})
    cats   = list({e["category"]     for e in entries})
    layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    for e in entries:
        layer_counts[e.get("layer", "L1")] += 1
    return {
        "uid": uid,
        "count": len(entries),
        "memories": sorted(entries, key=lambda x: x.get("ts",""), reverse=True),
        "available_models": models,
        "available_categories": cats,
        "layers": layer_counts,
    }

@app.get("/api/context-log/{uid}")
async def get_context_log(uid: str):
    return {
        "uid": uid,
        "log": list(reversed(context_log.get(uid, []))),  # newest first
    }

# ── Search + context (UID-less) ────────────────────────────────────────────
@app.post("/api/search")
async def search_local(request: Request):
    return await search_memories(_local_uid(), request)

@app.post("/api/context")
async def get_context(request: Request):
    """Return memories as a natural-language system instruction for LLM injection.
    Called by the browser-side fetch interceptor.

    Trusts the PhaseMemoryEngine for ALL retrieval logic:
      - Relevance scoring (Thermodynamic Resonance Retrieval)
      - Phase decay (gas items naturally fade out)
      - Contradiction detection (overridden facts suppressed)
      - Schema-aware query expansion
    No external filters needed — the engine IS the filter.
    """
    body = await request.json()
    query = body.get("query", "").strip()
    if not query or len(query) < 4:
        return {"context": "", "count": 0}

    MIN_RELEVANCE = 0.15  # Engine scores are well-calibrated; this just drops noise floor
    scope_uid = body.get("uid")
    if isinstance(scope_uid, str) and scope_uid.strip():
        # Extension / explicit tests send uid — scope retrieval to that profile only.
        mems = list(engine.search(query, scope_uid.strip(), limit=8))
    else:
        # No uid (e.g. daemon): search across all namespaces on this machine
        mems = []
        for ns in list(engine._items.keys()):
            mems.extend(engine.search(query, ns, limit=5))

    # Filter by engine's relevance score, sort descending
    mems = [(s, item) for s, item in mems if s >= MIN_RELEVANCE]
    mems.sort(key=lambda x: x[0], reverse=True)
    if not mems:
        return {"context": "", "count": 0}

    # Use the engine's own structured facts — no re-parsing needed
    facts = []
    seen = set()
    for score, item in mems:
        # The engine already extracted subject/relation/value at store time
        fact_text = item.fact.raw_text.strip()
        if not fact_text or len(fact_text) < 8:
            continue
        key = fact_text.lower()
        if key in seen:
            continue
        seen.add(key)
        print(f"[context] score={score:.3f} phase={_item_phase(item)} "
              f"s={item.fact.subject} r={item.fact.relation} → {fact_text[:60]}")
        facts.append(fact_text)
        if len(facts) >= 5:
            break

    if not facts:
        return {"context": "", "count": 0}

    # Natural system instruction — the LLM sees this as authoritative context
    ctx = (
        "The user has shared the following about themselves in previous conversations. "
        "Use this as background context — do not repeat it back unless asked:\n"
        + "\n".join(f"- {f}" for f in facts)
    )
    return {"context": ctx, "count": len(facts)}

@app.post("/api/search/{uid}")
async def search_memories(uid: str, request: Request):
    body  = await request.json()
    query = body.get("query", "").strip()
    limit = int(body.get("limit", 6))
    if not query:
        entries = list(reversed(memory_log.get(uid, [])))[:limit]
        return {"memories": entries, "count": len(entries)}
    mems = engine.search(query, uid, limit=limit)
    id_map = {e["id"]: e for e in memory_log.get(uid, [])}
    entries = []
    for _, item in mems:
        if item.id in id_map:
            entries.append(id_map[item.id])
        else:
            entries.append({"id": item.id, "text": item.fact.raw_text, "category": "General"})
    return {"memories": entries, "count": len(entries)}


# ── Daemon: store a message without calling external AI ─────────────────────
@app.post("/api/store/{uid}")
async def store_memory(uid: str, request: Request):
    body   = await request.json()
    text   = body.get("text", "").strip()
    source = body.get("source", "user")
    model  = body.get("model", "unknown")
    if not text:
        return JSONResponse({"error": "text required"}, 400)
    entry = await _store(uid, text, model, source)
    return {"ok": True, "entry": entry}


@app.delete("/api/memories/{uid}")
async def clear_memories(uid: str):
    memory_log.pop(uid, None)
    context_log.pop(uid, None)
    engine._items.pop(uid, None)
    engine._token_index.pop(uid, None)
    return {"ok": True}

def _local_uid() -> str:
    """Single source of truth: read or create ~/.clspp_uid."""
    uid_path = os.path.expanduser("~/.clspp_uid")
    if os.path.exists(uid_path):
        return open(uid_path).read().strip()
    uid = "u_" + hashlib.md5(os.path.expanduser("~").encode()).hexdigest()[:14]
    with open(uid_path, "w") as f:
        f.write(uid)
    return uid

@app.get("/api/uid")
async def get_uid():
    return {"uid": _local_uid()}

@app.get("/api/phase-stats")
async def phase_stats_local():
    return await get_phase_stats(_local_uid())

@app.get("/api/phase-stats/{uid}")
async def get_phase_stats(uid: str):
    """Return thermodynamic layer breakdown from PhaseMemoryEngine."""
    items = engine._items.get(uid, [])
    layers = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    phases = {"gas": 0, "liquid": 0, "solid": 0, "glass": 0}
    for item in items:
        p = _item_phase(item)
        phases[p] += 1
        layers[_phase_layer(p)] += 1

    # Also get debug info if available
    debug = {}
    if hasattr(engine, 'get_phase_debug'):
        try: debug = engine.get_phase_debug(uid)
        except: pass

    return {
        "uid": uid,
        "total": len(items),
        "layers": layers,
        "phases": phases,
        "debug": {
            "avg_strength": round(sum(i.consolidation_strength for i in items) / max(len(items), 1), 3),
            "avg_tau": round(sum(getattr(i, 'tau', 0) for i in items) / max(len(items), 1), 2),
            "schemas": layers["L2"] + layers["L3"],
        },
    }

# ── UID-less convenience routes (auto-resolve local user) ────────────────
@app.get("/api/memories")
async def get_memories_local(model: str = "", category: str = ""):
    return await get_memories(_local_uid(), model, category)

@app.get("/api/context-log")
async def get_context_log_local():
    return await get_context_log(_local_uid())

@app.websocket("/ws/memories")
async def ws_memories_local(ws: WebSocket):
    await ws_memories(ws, _local_uid())

@app.get("/health")
async def health():
    return {"status": "ok", "namespaces": len(memory_log),
            "total_memories": sum(len(v) for v in memory_log.values())}

@app.websocket("/ws/memories/{uid}")
async def ws_memories(ws: WebSocket, uid: str):
    await ws.accept()
    _ws_clients[uid].append(ws)
    for e in memory_log.get(uid, [])[-20:]:
        await ws.send_text(json.dumps({"type": "memory", **e}))
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect:
        if ws in _ws_clients[uid]: _ws_clients[uid].remove(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False, log_level="info")

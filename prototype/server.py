"""
CLS++ Memory Proxy Server — with Model Tracking + Classification

DEPRECATED: All routes have been merged into src/clsplusplus/local_routes.py
and are served by the unified server (python3 -m clsplusplus.main).
This file is kept for reference and for the standalone installer.
"""
import sys, os, json, re, hashlib, time as _time, asyncio
from collections import defaultdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from clsplusplus.memory_phase import PhaseMemoryEngine

engine = PhaseMemoryEngine()

# ── Persistence ───────────────────────────────────────────────────────────
_PERSIST_PATH = os.path.join(os.path.expanduser("~"), ".clspp", "memories.json")

def _save_to_disk():
    """Persist memory_log to disk so memories survive restarts."""
    try:
        os.makedirs(os.path.dirname(_PERSIST_PATH), exist_ok=True)
        with open(_PERSIST_PATH, "w") as f:
            json.dump(dict(memory_log), f)
    except Exception:
        pass

def _load_from_disk():
    """Reload memories from disk on startup."""
    if not os.path.exists(_PERSIST_PATH):
        return
    try:
        with open(_PERSIST_PATH) as f:
            data = json.load(f)
        for uid, entries in data.items():
            for e in entries:
                text = e.get("text", "")
                if not text or len(text.strip()) < 6:
                    continue
                # Ensure ts is set (fix "Invalid Date" in UI)
                if not e.get("ts"):
                    e["ts"] = datetime.utcnow().isoformat()
                item = engine.store(text, uid)
                if item:
                    # Update entry with fresh item ID but keep original metadata
                    e["id"] = item.id
                    memory_log[uid].append(e)
    except Exception:
        pass

# ── Extended memory store (wraps engine items with metadata) ──────────────
# { uid: [ {id, text, category, source_model, ts, strength} ] }
memory_log: dict[str, list[dict]] = defaultdict(list)

# ── Context bridge log ─────────────────────────────────────────────────────
# { uid: [ {model, memories_sent:[text], query, ts} ] }
context_log: dict[str, list[dict]] = defaultdict(list)

# ── WebSocket clients ──────────────────────────────────────────────────────
_ws_clients: dict[str, list[WebSocket]] = defaultdict(list)

# ── Extension activity tracker ─────────────────────────────────────────────
_extension_last_seen: Optional[float] = None

OPENAI_API_KEY    = os.getenv("CLS_OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("CLS_ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY    = os.getenv("CLS_GOOGLE_API_KEY", "")
OPENAI_BASE       = "https://api.openai.com"
ANTHROPIC_BASE    = "https://api.anthropic.com"

# ── Canonical context prompt ──────────────────────────────────────────────
CONTEXT_PREFIX = (
    "[MEMORY — VERIFIED USER FACTS]\n"
    "These are confirmed facts about this user from their own prior statements. "
    "Treat them as ground truth. If the user's current message contradicts a stored fact, "
    "gently remind them of what they previously said. Always prefer these facts over assumptions:"
)

app = FastAPI(title="CLS++ Memory Proxy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/ui", StaticFiles(directory=os.path.dirname(__file__), html=True), name="ui")

@app.on_event("startup")
async def _startup_load():
    _load_from_disk()


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
    clean = [(s, i) for s, i in memories if not i.fact.raw_text.strip().startswith("[Schema:")]
    if not clean: return messages
    block = CONTEXT_PREFIX + "\n" + "\n".join(f"- {i.fact.raw_text}" for _, i in clean)
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
    skip = ["i don't", "i do not", "however", "if you'd like", "let me know",
            "feel free", "i can help", "is there anything"]
    facts = []
    for s in sentences:
        sl = s.lower().strip()
        if len(sl) < 20: continue
        if not any(k in sl for k in kw): continue
        if any(k in sl for k in skip): continue
        facts.append(s.strip())
        if len(facts) >= 3: break
    return facts

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

_NOISE = [
    "personalization in progress", "connecting…", "try again",
    "the following facts were learned", "use this as background context",
    "the user has shared the following", "to monitor your current usage",
    "[cls++ memory]", "[schema:", "[memory — verified",
    "treat them as ground truth", "verified user facts",
    "how can i help you", "is there anything else", "let me know if",
    "do you have any", "feel free to", "i'd be happy to",
    "i don't have details", "i don't have specific", "i don't have information",
    "if you'd like to share", "i can help you with",
    "what would you like", "is there something specific",
]

def _is_question_only(text: str) -> bool:
    """Detect pure questions with no factual content worth storing."""
    t = text.strip().lower()
    q_starts = ["what is my", "what's my", "where do i", "where am i",
                "who am i", "do you know my", "tell me my", "what do you know",
                "can you tell me", "do you remember"]
    return any(t.startswith(q) for q in q_starts) and len(t) < 80

def _normalize_model(m: str) -> str:
    """Collapse model variants to display names: gemini-2.0-flash → gemini."""
    ml = m.lower()
    if "gemini" in ml: return "gemini"
    if "gpt" in ml or "chatgpt" in ml or "openai" in ml: return "chatgpt"
    if "claude" in ml or "anthropic" in ml: return "claude"
    if "copilot" in ml: return "copilot"
    return m

async def _store(uid: str, text: str, model: str, source: str = "user"):
    model = _normalize_model(model)
    t = text.strip()
    if len(t) < 6: return None
    tl = t.lower()
    if any(n in tl for n in _NOISE): return None
    if _is_question_only(t): return None
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
    asyncio.get_event_loop().call_soon(lambda: asyncio.ensure_future(asyncio.to_thread(_save_to_disk)))

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
        mem_block = CONTEXT_PREFIX + "\n" + "\n".join(f"- {i.fact.raw_text}" for _, i in mems)
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


# ── Gemini helper ─────────────────────────────────────────────────────────
def _gemini_call(system: str, user_msg: str, model_name: str = "gemini-2.0-flash") -> str:
    """Synchronous Gemini call — run via asyncio.to_thread."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        gm = genai.GenerativeModel(model_name, system_instruction=system)
        resp = gm.generate_content(user_msg)
        return resp.text or "No response from Gemini."
    except Exception as e:
        return f"Gemini error: {e}"


# ── Demo chat (powers memory.html live chat) ────────────────────────────────
@app.post("/api/chat/{uid}")
async def demo_chat(uid: str, request: Request):
    body    = await request.json()
    message = body.get("message","").strip()
    model   = body.get("model", "claude-haiku-4-5")
    if not message: return JSONResponse({"error":"message required"}, 400)

    # For small memory sets (<= 15 items), inject ALL facts for 100% recall.
    # For larger sets, use search-based retrieval.
    all_items = [i for i in (engine._items.get(uid, []))
                 if not i.fact.raw_text.strip().startswith("[Schema:")]
    if len(all_items) <= 100:
        mems = [(1.0, i) for i in all_items]
    else:
        mems = engine.search(message, uid, limit=10)
        mems = [(s, i) for s, i in mems if not i.fact.raw_text.strip().startswith("[Schema:")]
    mems = mems[:40]  # Modern LLMs handle 40 facts easily (~2K tokens)
    _log_context(uid, model, message, mems)
    await _store(uid, message, model, "user")

    system = "You are a helpful AI assistant. Be concise."
    if mems:
        mem_block = CONTEXT_PREFIX + "\n" + "\n".join(f"- {i.fact.raw_text}" for _, i in mems)
        system = mem_block + "\n\n" + system

    reply = "Error reaching AI."
    try:
        if model.startswith("gemini"):
            # ── Gemini route ──────────────────────────────────────────
            reply = await asyncio.to_thread(_gemini_call, system, message, model)
        else:
            async with httpx.AsyncClient(timeout=60) as client:
                if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
                    # ── OpenAI route ──────────────────────────────────────
                    resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions",
                        json={"model": model, "max_tokens": 512,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user",   "content": message}]},
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                 "Content-Type": "application/json"})
                    reply = resp.json()["choices"][0]["message"]["content"]
                else:
                    # ── Anthropic route ───────────────────────────────────
                    resp = await client.post(f"{ANTHROPIC_BASE}/v1/messages",
                        json={"model": model, "max_tokens": 512, "system": system,
                              "messages": [{"role": "user", "content": message}]},
                        headers={"x-api-key": ANTHROPIC_API_KEY,
                                 "anthropic-version": "2023-06-01",
                                 "Content-Type": "application/json"})
                    reply = resp.json()["content"][0]["text"]
    except Exception as e:
        reply = f"Error reaching AI: {e}"

    # Don't store AI replies in demo chat — they pollute the memory store
    # with noise like "I don't have information" and push real facts down.
    # User facts are already stored above.

    return {
        "reply": reply,
        "memories_used": len(mems),
        "memory_snippets": [i.fact.raw_text for _, i in mems],
        "model": model,
    }


# ── REST: memories with full metadata ─────────────────────────────────────

@app.get("/api/memories/{uid}")
async def get_memories(uid: str, model: str = "", category: str = "",
                       label: str = "", layer: str = "",
                       sort: str = "ts", order: str = "desc"):
    """Return all memories for the UI viewer with filtering and sorting."""
    items = engine._items.get(uid, [])
    log_map = {e["id"]: e for e in memory_log.get(uid, [])}

    entries = []
    for item in items:
        phase = _item_phase(item)
        if phase == "gas":
            continue
        # Skip raw schema items from display — internal engine artifacts
        if item.fact.raw_text.strip().startswith("[Schema:"):
            continue
        log_entry = log_map.get(item.id, {})
        entry = {
            "id": item.id,
            "text": item.fact.raw_text,
            "subject": item.fact.subject,
            "relation": item.fact.relation,
            "value": item.fact.value,
            "category": log_entry.get("category", classify(item.fact.raw_text)),
            "source_model": _normalize_model(log_entry.get("source_model", "unknown")),
            "source": log_entry.get("source", "user"),
            "strength": round(item.consolidation_strength, 3),
            "phase": phase,
            "layer": _phase_layer(phase),
            "tau": round(getattr(item, 'tau', 0), 2),
            "retrieval_count": getattr(item, 'retrieval_count', 0),
            "labels": log_entry.get("labels", []),
            "ts": log_entry.get("ts", ""),
        }
        entries.append(entry)

    if model:    entries = [e for e in entries if e["source_model"] == model]
    if category: entries = [e for e in entries if e["category"] == category]
    if label:    entries = [e for e in entries if label in e.get("labels", [])]
    if layer:    entries = [e for e in entries if e.get("layer") == layer]

    sort_keys = {
        "ts":       lambda x: x.get("ts", ""),
        "strength": lambda x: x.get("strength", 0),
        "category": lambda x: x.get("category", ""),
        "model":    lambda x: x.get("source_model", ""),
        "phase":    lambda x: {"gas":0,"liquid":1,"solid":2,"glass":3}.get(x.get("phase","liquid"),1),
    }
    key_fn = sort_keys.get(sort, sort_keys["ts"])
    entries.sort(key=key_fn, reverse=(order == "desc"))

    models = list({e["source_model"] for e in entries})
    cats   = list({e["category"]     for e in entries})
    all_labels = sorted({l for e in entries for l in e.get("labels", [])})
    layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
    for e in entries:
        layer_counts[e.get("layer", "L1")] += 1
    return {
        "uid": uid,
        "count": len(entries),
        "memories": entries,
        "available_models": models,
        "available_categories": cats,
        "available_labels": all_labels,
        "layers": layer_counts,
    }

@app.patch("/api/memories/{uid}/{memory_id}/labels")
async def update_labels(uid: str, memory_id: str, request: Request):
    """Add or update labels on a memory."""
    body = await request.json()
    labels = [str(l).strip()[:30] for l in body.get("labels", [])[:10] if str(l).strip()]
    for entry in memory_log.get(uid, []):
        if entry.get("id") == memory_id:
            entry["labels"] = labels
            _save_to_disk()
            return {"ok": True, "labels": labels}
    # Also check all UIDs (merged namespace case)
    for u, entries in memory_log.items():
        for entry in entries:
            if entry.get("id") == memory_id:
                entry["labels"] = labels
                _save_to_disk()
                return {"ok": True, "labels": labels}
    return JSONResponse({"error": "memory not found"}, 404)

@app.get("/api/context-log")
async def get_context_log_local():
    return await get_context_log(_local_uid())

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
    uid = body.get("uid") or _local_uid()
    query = body.get("query", "").strip()
    if not query or len(query) < 4:
        return {"context": "", "count": 0}

    # For small memory sets, inject ALL facts for perfect recall at launch.
    # For larger sets, use search-based retrieval.
    all_items = []
    for ns in list(engine._items.keys()):
        all_items.extend([(1.0, i) for i in engine._items.get(ns, [])
                          if not i.fact.raw_text.strip().startswith("[Schema:")])

    if len(all_items) <= 100:
        mems = all_items
    else:
        mems = []
        for ns in list(engine._items.keys()):
            mems.extend(engine.search(query, ns, limit=8))
        mems = [(s, item) for s, item in mems
                if s > 0.001 and not item.fact.raw_text.strip().startswith("[Schema:")]
        mems.sort(key=lambda x: x[0], reverse=True)
    mems = mems[:30]  # Up to 30 facts in extension context injection
    if not mems:
        return {"context": "", "count": 0}

    # Use the engine's own structured facts — no re-parsing needed
    facts = []
    seen = set()
    for score, item in mems:
        fact_text = item.fact.raw_text.strip()
        if not fact_text or len(fact_text) < 8:
            continue
        # Skip raw schema representations — they confuse LLMs
        if fact_text.startswith("[Schema:"):
            continue
        key = fact_text.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact_text)
        if len(facts) >= 30:
            break

    if not facts:
        return {"context": "", "count": 0}

    ctx = CONTEXT_PREFIX + "\n" + "\n".join(f"- {f}" for f in facts)

    global _extension_last_seen
    _extension_last_seen = _time.time()

    # Log to context_log so the UI Context Log panel shows injection history
    context_log[uid].append({
        "model": "extension",
        "query": query[:120],
        "memories_sent": facts,
        "count": len(facts),
        "ts": datetime.utcnow().isoformat(),
    })

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
async def get_memories_local(model: str = "", category: str = "", label: str = "",
                             layer: str = "", sort: str = "ts", order: str = "desc"):
    result = await get_memories(_local_uid(), model, category, label, layer, sort, order)
    if result["count"] == 0 and engine._items:
        all_entries = []
        all_labels_set = set()
        for uid in list(engine._items.keys()):
            r = await get_memories(uid, model, category, label, layer, sort, order)
            all_entries.extend(r.get("memories", []))
            all_labels_set.update(r.get("available_labels", []))
        models = list({e["source_model"] for e in all_entries})
        cats = list({e["category"] for e in all_entries})
        layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        for e in all_entries:
            layer_counts[e.get("layer", "L1")] += 1
        return {
            "uid": "all",
            "count": len(all_entries),
            "memories": all_entries,
            "available_models": models,
            "available_categories": cats,
            "available_labels": sorted(all_labels_set),
            "layers": layer_counts,
        }
    return result

@app.get("/api/context-log")
async def get_context_log_local():
    return await get_context_log(_local_uid())

@app.websocket("/ws/memories")
async def ws_memories_local(ws: WebSocket):
    await ws_memories(ws, _local_uid())

@app.get("/")
async def root():
    return RedirectResponse(url="/ui/index.html")

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

# ── Install API — in-browser installer ────────────────────────────────────

import shutil
import subprocess as _sp
from pathlib import Path

_INSTALL_DIR = Path.home() / ".clspp"
_SRC_DIR = Path(__file__).parent          # prototype/
_ENGINE_DIR = _SRC_DIR.parent / "src"     # src/
_EXT_DIR = _SRC_DIR.parent / "extension"  # extension/

@app.get("/api/install/status")
async def install_status():
    installed = (_INSTALL_DIR / "server.py").exists()
    daemon_running = False
    try:
        r = _sp.run(["pgrep", "-f", "daemon.py"], capture_output=True, timeout=2)
        daemon_running = r.returncode == 0
    except Exception:
        pass
    return {
        "installed": installed,
        "daemon_running": daemon_running,
        "install_dir": str(_INSTALL_DIR),
        "os": "macos",
    }

@app.post("/api/install/run")
async def install_run():
    """Server-side install: copy files, install deps, set up LaunchAgent.
    Returns streaming JSON lines for progress updates."""
    import asyncio

    async def _stream():
        steps = []
        try:
            # Step 1: Create directory
            yield json.dumps({"step": 1, "msg": "Creating install directory..."}) + "\n"
            _INSTALL_DIR.mkdir(parents=True, exist_ok=True)
            steps.append("directory")

            # Step 2: Copy engine
            yield json.dumps({"step": 2, "msg": "Copying memory engine..."}) + "\n"
            dst_engine = _INSTALL_DIR / "engine" / "clsplusplus"
            if dst_engine.exists():
                shutil.rmtree(dst_engine)
            shutil.copytree(_ENGINE_DIR / "clsplusplus", dst_engine)
            steps.append("engine")

            # Step 3: Copy server + daemon + UI
            yield json.dumps({"step": 3, "msg": "Copying server and UI files..."}) + "\n"
            for f in ["server.py", "daemon.py", "memory.html", "index.html"]:
                src = _SRC_DIR / f
                if src.exists():
                    shutil.copy2(src, _INSTALL_DIR / f)
            # Patch server.py paths for installed location
            srv = _INSTALL_DIR / "server.py"
            if srv.exists():
                txt = srv.read_text()
                txt = txt.replace(
                    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))",
                    "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))"
                )
                txt = txt.replace(
                    "load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))",
                    "load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))"
                )
                srv.write_text(txt)
            # Copy .env if exists
            env_src = _SRC_DIR.parent / ".env"
            if env_src.exists():
                shutil.copy2(env_src, _INSTALL_DIR / ".env")
            steps.append("files")

            # Step 4: Copy Chrome extension
            yield json.dumps({"step": 4, "msg": "Bundling Chrome extension..."}) + "\n"
            dst_ext = _INSTALL_DIR / "extension"
            if dst_ext.exists():
                shutil.rmtree(dst_ext)
            shutil.copytree(_EXT_DIR, dst_ext)
            steps.append("extension")

            # Step 5: Install Python dependencies
            yield json.dumps({"step": 5, "msg": "Installing dependencies (this may take a minute)..."}) + "\n"
            deps = [
                "fastapi", "uvicorn[standard]", "httpx", "python-dotenv", "requests",
                "rumps", "pyobjc-framework-Quartz>=10.0", "pyobjc-framework-ApplicationServices>=10.0",
            ]
            proc = _sp.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + deps,
                capture_output=True, text=True, timeout=120
            )
            steps.append("deps")

            # Step 6: Create launch.sh
            yield json.dumps({"step": 6, "msg": "Creating launcher..."}) + "\n"
            launch_sh = _INSTALL_DIR / "launch.sh"
            launch_sh.write_text(f"""#!/bin/bash
DIR="$HOME/.clspp"
LOG="$DIR/.clspp.log"
lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null
sleep 0.2
PYTHONPATH="$DIR/engine" python3 "$DIR/server.py" >> "$LOG" 2>&1 &
for i in $(seq 1 20); do
  curl -s http://localhost:8080/health > /dev/null 2>&1 && break
  sleep 0.5
done
python3 "$DIR/daemon.py" >> "$LOG" 2>&1 &
""")
            launch_sh.chmod(0o755)
            steps.append("launcher")

            # Step 7: Create LaunchAgent plist
            yield json.dumps({"step": 7, "msg": "Setting up auto-start on login..."}) + "\n"
            plist_dir = Path.home() / "Library" / "LaunchAgents"
            plist_dir.mkdir(parents=True, exist_ok=True)
            plist = plist_dir / "com.clspp.daemon.plist"
            plist.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.clspp.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{_INSTALL_DIR}/launch.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{_INSTALL_DIR}/.clspp.log</string>
    <key>StandardErrorPath</key>
    <string>{_INSTALL_DIR}/.clspp.log</string>
</dict>
</plist>
""")
            steps.append("launchagent")

            yield json.dumps({"step": 8, "msg": "Done!", "ok": True, "steps": steps}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e), "steps": steps}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")

@app.get("/api/detect-browsers")
async def detect_browsers():
    """Detect installed Chromium-based browsers for the activation page."""
    import platform
    browsers = [
        {"id": "chrome",  "apple": "Google Chrome",  "path": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"},
        {"id": "brave",   "apple": "Brave Browser",  "path": "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"},
        {"id": "arc",     "apple": "Arc",             "path": "/Applications/Arc.app/Contents/MacOS/Arc"},
        {"id": "edge",    "apple": "Microsoft Edge",  "path": "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"},
        {"id": "chromium","apple": "Chromium",        "path": "/Applications/Chromium.app/Contents/MacOS/Chromium"},
    ]
    # Also check ~/Applications
    home_apps = os.path.expanduser("~/Applications")
    for b in list(browsers):
        alt = os.path.join(home_apps, f"{b['apple']}.app", "Contents", "MacOS", b["apple"])
        if not os.path.exists(b["path"]) and os.path.exists(alt):
            b["path"] = alt
    found = [b for b in browsers if os.path.exists(b["path"])]
    ext_dir = str(_EXT_DIR) if _EXT_DIR.is_dir() else None
    return {
        "ok": len(found) > 0,
        "browsers": found,
        "os": platform.system().lower().replace("darwin", "darwin"),
        "extension_dir": ext_dir,
    }

@app.get("/api/extension-status")
async def extension_status():
    """Returns whether the CLS++ Chrome extension has recently made API calls."""
    import time as _time
    loaded = _extension_last_seen is not None and (_time.time() - _extension_last_seen) < 300
    return {"loaded": loaded, "last_seen": _extension_last_seen}

@app.post("/api/install/start")
async def install_start():
    """Start the daemon process."""
    try:
        daemon_path = _INSTALL_DIR / "daemon.py" if (_INSTALL_DIR / "daemon.py").exists() else _SRC_DIR / "daemon.py"
        proc = _sp.Popen(
            [sys.executable, str(daemon_path)],
            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
            start_new_session=True
        )
        return {"ok": True, "pid": proc.pid}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/api/install/browser-check")
async def browser_check():
    """Return which Chromium-based browser is installed, if any."""
    browsers = [
        ("Google Chrome",  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ("Brave Browser",  "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        ("Arc",            "/Applications/Arc.app/Contents/MacOS/Arc"),
        ("Microsoft Edge", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ("Chromium",       "/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]
    for name, path in browsers:
        if os.path.exists(path):
            return {"found": True, "browser": name}
    return {"found": False, "browser": None}

@app.post("/api/install/launch-browser")
async def launch_browser():
    """Quit the user's Chromium browser and relaunch it with the CLS++ extension loaded."""
    import asyncio

    ext_path = str(_INSTALL_DIR / "extension")
    if not (_INSTALL_DIR / "extension" / "manifest.json").exists():
        ext_path = str(_EXT_DIR)

    memory_url = "http://localhost:8080/ui/memory.html"

    # Ordered by popularity; Arc uses Chromium so --load-extension works
    browsers = [
        ("Google Chrome",    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ("Brave Browser",    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        ("Arc",              "/Applications/Arc.app/Contents/MacOS/Arc"),
        ("Microsoft Edge",   "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        ("Chromium",         "/Applications/Chromium.app/Contents/MacOS/Chromium"),
    ]

    launched = None
    for name, path in browsers:
        if not os.path.exists(path):
            continue
        # Quit gracefully so Chrome saves the session
        _sp.run(["osascript", "-e", f'tell application "{name}" to quit'],
                capture_output=True, timeout=6)
        await asyncio.sleep(2)
        _sp.Popen([path,
                   f"--load-extension={ext_path}",
                   "--no-first-run",
                   memory_url])
        launched = name
        break

    if not launched:
        # No Chromium-based browser — just open the memory page in the default browser
        _sp.Popen(["open", memory_url])
        return {"ok": False, "browser": None,
                "message": "No Chrome/Brave/Arc found. Opened memory page in default browser."}

    return {"ok": True, "browser": launched}

# ── Mock echo endpoints for E2E injection testing ────────────────────────────
# The Chrome extension intercept.js rewrites fetch bodies so the AI site's
# request includes CLS++ context.  These mock endpoints validate that the
# rewritten body still has the expected shape and that the memory text is present.

@app.post("/backend-api/conversation")
async def mock_chatgpt_echo(request: Request):
    """Mock ChatGPT backend — checks that CLS++ injection is present."""
    body = await request.json()
    messages = body.get("messages", [])
    injection_ok = False
    for m in messages:
        content = m.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        for p in parts:
            if isinstance(p, str) and "previous conversations" in p.lower():
                injection_ok = True
                break
    return {"injection_ok": injection_ok, "messages_count": len(messages)}


@app.post("/api/e2e/chat_conversations/{conversation_id}/completion")
async def mock_claude_echo(conversation_id: str, request: Request):
    """Mock Claude API — checks that CLS++ injection is present."""
    body = await request.json()
    prompt = body.get("prompt", "")
    injection_ok = "previous conversations" in prompt.lower()
    return {"injection_ok": injection_ok, "conversation_id": conversation_id}


# ── macOS installer routes ───────────────────────────────────────────────────

_DOWNLOADS_DIR = Path(__file__).resolve().parent.parent / "downloads"


def _workspace_repo_with_engine() -> Optional[str]:
    """Return project root if we're running inside the dev repo with the engine."""
    root = Path(__file__).resolve().parent.parent
    if (root / "src" / "clsplusplus" / "memory_phase.py").is_file():
        return str(root)
    return None


@app.get("/install/macos")
async def install_macos_download():
    """Serve the macOS installer zip if it exists in downloads/."""
    zips = sorted(_DOWNLOADS_DIR.glob("*.zip")) if _DOWNLOADS_DIR.is_dir() else []
    if not zips:
        return JSONResponse({"error": "no_installer"}, status_code=404)
    latest = zips[-1]
    return Response(
        content=latest.read_bytes(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{latest.name}"'},
    )


@app.get("/install/macos/status")
async def install_macos_status():
    """Return current install phase."""
    installed = (_INSTALL_DIR / "server.py").exists()
    daemon_running = False
    try:
        r = _sp.run(["pgrep", "-f", "daemon.py"], capture_output=True, timeout=2)
        daemon_running = r.returncode == 0
    except Exception:
        pass
    phase = "running" if daemon_running else ("installed" if installed else "fresh")
    return {"phase": phase, "installed": installed, "daemon_running": daemon_running}


@app.post("/install/macos/apply")
async def install_macos_apply():
    """One-click install — only allowed when running from the dev repo."""
    repo = _workspace_repo_with_engine()
    if not repo:
        return JSONResponse({"error": "not_in_dev_repo"}, status_code=404)
    # Delegate to the existing install flow
    return JSONResponse({"ok": True, "repo": repo, "message": "Use /api/install/run for full install."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=False, log_level="info")

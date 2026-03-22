"""
CLS++ Memory Proxy Server — with Model Tracking + Classification
"""
import sys, os, json, re, hashlib
from collections import defaultdict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, JSONResponse
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

async def _store(uid: str, text: str, model: str, source: str = "user"):
    if len(text.strip()) < 6: return None
    item = engine.store(text, uid)
    if item is None: return None
    cat = classify(text)
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

    system = "You are a helpful AI assistant with memory about this user."
    if mems:
        system = "[What you remember about this user]\n" + \
                 "\n".join(f"• {i.fact.raw_text}" for _, i in mems) + "\n\n" + system

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{ANTHROPIC_BASE}/v1/messages",
            json={"model": model, "max_tokens": 512, "system": system,
                  "messages": [{"role": "user", "content": message}]},
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01", "Content-Type": "application/json"})

    try:    reply = resp.json()["content"][0]["text"]
    except: reply = "Error reaching AI."

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
    entries = memory_log.get(uid, [])
    if model:    entries = [e for e in entries if e["source_model"] == model]
    if category: entries = [e for e in entries if e["category"] == category]
    models = list({e["source_model"] for e in memory_log.get(uid, [])})
    cats   = list({e["category"]     for e in memory_log.get(uid, [])})
    return {
        "uid": uid,
        "count": len(entries),
        "memories": list(reversed(entries)),   # newest first
        "available_models": models,
        "available_categories": cats,
    }

@app.get("/api/context-log/{uid}")
async def get_context_log(uid: str):
    return {
        "uid": uid,
        "log": list(reversed(context_log.get(uid, []))),  # newest first
    }

@app.delete("/api/memories/{uid}")
async def clear_memories(uid: str):
    memory_log.pop(uid, None)
    context_log.pop(uid, None)
    engine._items.pop(uid, None)
    engine._token_index.pop(uid, None)
    return {"ok": True}

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

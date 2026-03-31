"""
CLS++ Local / Daemon Routes — ported from prototype/server.py

Provides browser extension support, memory viewer API, LLM proxies,
WebSocket streaming, and macOS installer routes. Registered as an
APIRouter in create_app().
"""
import hashlib
import json
import os
import re
import shutil
import subprocess as _sp
import sys
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse

from clsplusplus.config import Settings
from clsplusplus.memory_service import MemoryService


# ── Classification ─────────────────────────────────────────────────────────
CATEGORIES = {
    "Identity":     r"\b(name|called|i am|i'm|born|from|live|based|nationality|age|old)\b",
    "Preference":   r"\b(prefer|like|love|hate|enjoy|dislike|favourite|favorite|don't like|want)\b",
    "Work":         r"\b(work|job|role|company|employer|position|career|profession|office|salary)\b",
    "Project":      r"\b(build|building|startup|product|project|launch|app|platform|startup|mvp)\b",
    "Relationship": r"\b(wife|husband|partner|friend|colleague|team|family|brother|sister|son|daughter|manager|boss)\b",
    "Goal":         r"\b(goal|plan|want to|trying|aim|vision|mission|target|milestone)\b",
    "Temporal":     r"\b(yesterday|today|tomorrow|last week|next week|deadline|schedule|meeting|remind)\b",
    "Context":      r"\b(remember|note|context|background|mentioned|told|said|asked|discussed)\b",
}


def classify(text: str) -> str:
    t = text.lower()
    for cat, pattern in CATEGORIES.items():
        if re.search(pattern, t):
            return cat
    return "General"


def _user_text(messages: list) -> str:
    parts = []
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        parts.append(b["text"])
    return " ".join(parts).strip()


def _inject(messages: list, memories: list) -> list:
    if not memories:
        return messages
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
        if hasattr(item.schema_meta, 'H_history') and len(item.schema_meta.H_history) >= 3:
            vals = item.schema_meta.H_history[-3:]
            mean = sum(vals) / len(vals)
            if mean > 0 and max(abs(v - mean) for v in vals) / mean < 0.01:
                return "glass"
        return "solid"
    return "liquid"


def _phase_layer(phase: str) -> str:
    return {"gas": "L0", "liquid": "L1", "solid": "L2", "glass": "L3"}.get(phase, "L1")


def _local_uid() -> str:
    """Single source of truth: read or create ~/.clspp_uid."""
    uid_path = os.path.expanduser("~/.clspp_uid")
    if os.path.exists(uid_path):
        return open(uid_path).read().strip()
    uid = "u_" + hashlib.md5(os.path.expanduser("~").encode()).hexdigest()[:14]
    with open(uid_path, "w") as f:
        f.write(uid)
    return uid


def create_local_router(memory_service: MemoryService, settings: Settings) -> APIRouter:
    """Create an APIRouter with all local/daemon routes."""
    router = APIRouter()
    engine = memory_service.engine

    # ── In-memory state (closure-scoped) ──────────────────────────────────
    memory_log: dict[str, list[dict]] = defaultdict(list)
    context_log: dict[str, list[dict]] = defaultdict(list)
    _ws_clients: dict[str, list[WebSocket]] = defaultdict(list)
    _extension_last_seen: dict[str, Optional[float]] = {"ts": None}  # mutable container

    OPENAI_API_KEY = getattr(settings, 'openai_api_key', '') or os.getenv("CLS_OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = getattr(settings, 'anthropic_api_key', '') or os.getenv("CLS_ANTHROPIC_API_KEY", "")
    OPENAI_BASE = "https://api.openai.com"
    ANTHROPIC_BASE = "https://api.anthropic.com"

    def _ns(request: Request, body: dict) -> str:
        return (request.headers.get("X-User-Id")
                or body.get("user")
                or hashlib.sha256(request.client.host.encode()).hexdigest()[:16])

    async def _store(uid: str, text: str, model: str, source: str = "user"):
        if len(text.strip()) < 6:
            return None
        item = engine.store(text, uid)
        if item is None:
            return None
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
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients[uid].remove(ws)
        return entry

    def _log_context(uid: str, model: str, query: str, memories: list):
        context_log[uid].append({
            "model": model,
            "query": query[:120],
            "memories_sent": [i.fact.raw_text for _, i in memories],
            "count": len(memories),
            "ts": datetime.utcnow().isoformat(),
        })

    # ── OpenAI proxy ───────────────────────────────────────────────────────
    @router.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        body = await request.json()
        uid = _ns(request, body)
        model_name = body.get("model", "openai")
        query = _user_text(body.get("messages", []))
        mems = engine.search(query, uid, limit=6) if query else []

        _log_context(uid, model_name, query, mems)
        body["messages"] = _inject(body.get("messages", []), mems)

        auth = request.headers.get("Authorization") or f"Bearer {OPENAI_API_KEY}"
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions",
                json=body, headers={"Authorization": auth, "Content-Type": "application/json"})

        if query:
            await _store(uid, query, model_name, "user")
        try:
            content = resp.json()["choices"][0]["message"]["content"]
            for f in _assistant_facts(content):
                await _store(uid, f, model_name, "assistant")
        except Exception:
            pass
        return Response(content=resp.content, media_type="application/json", status_code=resp.status_code)

    # ── Anthropic proxy ────────────────────────────────────────────────────
    @router.post("/v1/messages")
    async def anthropic_messages(request: Request):
        body = await request.json()
        uid = _ns(request, body)
        model_name = body.get("model", "claude")
        query = _user_text(body.get("messages", []))
        mems = engine.search(query, uid, limit=6) if query else []

        _log_context(uid, model_name, query, mems)
        if mems:
            mem_block = "[CLS++ Memory]\n" + "\n".join(f"• {i.fact.raw_text}" for _, i in mems)
            body["system"] = (mem_block + "\n\n" + body.get("system", "")).strip()

        auth_key = (request.headers.get("x-api-key")
                    or request.headers.get("Authorization", "").replace("Bearer ", "")
                    or ANTHROPIC_API_KEY)
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(f"{ANTHROPIC_BASE}/v1/messages", json=body,
                headers={"x-api-key": auth_key, "anthropic-version": "2023-06-01",
                         "Content-Type": "application/json"})

        if query:
            await _store(uid, query, model_name, "user")
        try:
            content = resp.json()["content"][0]["text"]
            for f in _assistant_facts(content):
                await _store(uid, f, model_name, "assistant")
        except Exception:
            pass
        return Response(content=resp.content, media_type="application/json", status_code=resp.status_code)

    # ── Demo chat (powers memory.html live chat) ────────────────────────────
    @router.post("/api/chat/{uid}")
    async def demo_chat(uid: str, request: Request):
        body = await request.json()
        message = body.get("message", "").strip()
        model = body.get("model", "claude-haiku-4-5")
        if not message:
            return JSONResponse({"error": "message required"}, 400)

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
                    resp = await client.post(f"{OPENAI_BASE}/v1/chat/completions",
                        json={"model": model, "max_tokens": 512,
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": message}]},
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                 "Content-Type": "application/json"})
                    reply = resp.json()["choices"][0]["message"]["content"]
                else:
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

    # ── REST: memories with full metadata ─────────────────────────────────
    @router.get("/api/memories/{uid}")
    async def get_memories(uid: str, model: str = "", category: str = ""):
        items = engine._items.get(uid, [])
        log_map = {e["id"]: e for e in memory_log.get(uid, [])}

        entries = []
        for item in items:
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

        if model:
            entries = [e for e in entries if e["source_model"] == model]
        if category:
            entries = [e for e in entries if e["category"] == category]
        models = list({e["source_model"] for e in entries})
        cats = list({e["category"] for e in entries})
        layer_counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        for e in entries:
            layer_counts[e.get("layer", "L1")] += 1
        return {
            "uid": uid,
            "count": len(entries),
            "memories": sorted(entries, key=lambda x: x.get("ts", ""), reverse=True),
            "available_models": models,
            "available_categories": cats,
            "layers": layer_counts,
        }

    @router.get("/api/memories")
    async def get_memories_local(model: str = "", category: str = ""):
        return await get_memories(_local_uid(), model, category)

    # ── Context log ────────────────────────────────────────────────────────
    @router.get("/api/context-log/{uid}")
    async def get_context_log(uid: str):
        return {
            "uid": uid,
            "log": list(reversed(context_log.get(uid, []))),
        }

    @router.get("/api/context-log")
    async def get_context_log_local():
        return await get_context_log(_local_uid())

    # ── Search + context ──────────────────────────────────────────────────
    @router.post("/api/search/{uid}")
    async def search_memories(uid: str, request: Request):
        body = await request.json()
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

    @router.post("/api/search")
    async def search_local(request: Request):
        return await search_memories(_local_uid(), request)

    @router.post("/api/context")
    async def get_context(request: Request):
        """Return memories as a natural-language system instruction for LLM injection.
        Called by the browser-side fetch interceptor."""
        body = await request.json()
        uid = body.get("uid") or _local_uid()
        query = body.get("query", "").strip()
        if not query or len(query) < 4:
            return {"context": "", "count": 0}

        MIN_RELEVANCE = 0.15
        mems = []
        for ns in list(engine._items.keys()):
            mems.extend(engine.search(query, ns, limit=5))

        mems = [(s, item) for s, item in mems if s >= MIN_RELEVANCE]
        mems.sort(key=lambda x: x[0], reverse=True)
        if not mems:
            return {"context": "", "count": 0}

        facts = []
        seen = set()
        for score, item in mems:
            fact_text = item.fact.raw_text.strip()
            if not fact_text or len(fact_text) < 8:
                continue
            key = fact_text.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact_text)
            if len(facts) >= 5:
                break

        if not facts:
            return {"context": "", "count": 0}

        ctx = (
            "The user has shared the following about themselves in previous conversations. "
            "Use this as background context — do not repeat it back unless asked:\n"
            + "\n".join(f"- {f}" for f in facts)
        )

        _extension_last_seen["ts"] = _time.time()

        context_log[uid].append({
            "model": "extension",
            "query": query[:120],
            "memories_sent": facts,
            "count": len(facts),
            "ts": datetime.utcnow().isoformat(),
        })

        return {"context": ctx, "count": len(facts)}

    # ── Store / Delete / UID ───────────────────────────────────────────────
    @router.post("/api/store/{uid}")
    async def store_memory(uid: str, request: Request):
        body = await request.json()
        text = body.get("text", "").strip()
        source = body.get("source", "user")
        model = body.get("model", "unknown")
        if not text:
            return JSONResponse({"error": "text required"}, 400)
        entry = await _store(uid, text, model, source)
        return {"ok": True, "entry": entry}

    @router.delete("/api/memories/{uid}")
    async def clear_memories(uid: str):
        memory_log.pop(uid, None)
        context_log.pop(uid, None)
        engine._items.pop(uid, None)
        engine._token_index.pop(uid, None)
        return {"ok": True}

    @router.get("/api/uid")
    async def get_uid():
        return {"uid": _local_uid()}

    # ── Phase stats ────────────────────────────────────────────────────────
    @router.get("/api/phase-stats/{uid}")
    async def get_phase_stats(uid: str):
        items = engine._items.get(uid, [])
        layers = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        phases = {"gas": 0, "liquid": 0, "solid": 0, "glass": 0}
        for item in items:
            p = _item_phase(item)
            phases[p] += 1
            layers[_phase_layer(p)] += 1

        debug = {}
        if hasattr(engine, 'get_phase_debug'):
            try:
                debug = engine.get_phase_debug(uid)
            except Exception:
                pass

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

    @router.get("/api/phase-stats")
    async def phase_stats_local():
        return await get_phase_stats(_local_uid())

    # ── WebSocket memory streams ──────────────────────────────────────────
    @router.websocket("/ws/memories/{uid}")
    async def ws_memories(ws: WebSocket, uid: str):
        await ws.accept()
        _ws_clients[uid].append(ws)
        for e in memory_log.get(uid, [])[-20:]:
            await ws.send_text(json.dumps({"type": "memory", **e}))
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            if ws in _ws_clients[uid]:
                _ws_clients[uid].remove(ws)

    @router.websocket("/ws/memories")
    async def ws_memories_local(ws: WebSocket):
        await ws_memories(ws, _local_uid())

    # ── Extension status ──────────────────────────────────────────────────
    @router.get("/api/extension-status")
    async def extension_status():
        ts = _extension_last_seen["ts"]
        loaded = ts is not None and (_time.time() - ts) < 300
        return {"loaded": loaded, "last_seen": ts}

    # ── Extension download — zip up the extension/ dir on demand ────────────
    _EXT_DIR_GLOBAL = Path(__file__).resolve().parent.parent.parent / "extension"

    @router.get("/extension/download")
    async def extension_download():
        """Package the Chrome extension as a zip and serve it for download."""
        import io, zipfile as _zipfile
        if not _EXT_DIR_GLOBAL.is_dir():
            return JSONResponse({"error": "extension_not_found"}, status_code=404)
        buf = io.BytesIO()
        with _zipfile.ZipFile(buf, "w", _zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(_EXT_DIR_GLOBAL.rglob("*")):
                if f.is_file() and ".git" not in f.parts and "node_modules" not in f.parts:
                    zf.write(f, f.relative_to(_EXT_DIR_GLOBAL.parent))
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="clsplusplus-extension.zip"'},
        )

    # ── Install API — local-only, disabled in cloud containers ─────────────
    if os.getenv("CLS_ENABLE_INSTALLER"):
        _INSTALL_DIR = Path.home() / ".clspp"
        _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        _ENGINE_DIR = _PROJECT_ROOT / "src"
        _EXT_DIR = _PROJECT_ROOT / "extension"
        _PROTO_DIR = _PROJECT_ROOT / "prototype"
        _WEBSITE_DIR = _PROJECT_ROOT / "website"

        @router.get("/api/install/status")
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
                "os": "macos" if sys.platform == "darwin" else sys.platform,
            }

        @router.post("/api/install/run")
        async def install_run():
            """Server-side install: copy files, install deps, set up LaunchAgent.
            Returns streaming JSON lines for progress updates."""
            import asyncio

            async def _stream():
                steps = []
                try:
                    yield json.dumps({"step": 1, "msg": "Creating install directory..."}) + "\n"
                    _INSTALL_DIR.mkdir(parents=True, exist_ok=True)
                    steps.append("directory")

                    yield json.dumps({"step": 2, "msg": "Copying memory engine..."}) + "\n"
                    dst_engine = _INSTALL_DIR / "engine" / "clsplusplus"
                    if dst_engine.exists():
                        shutil.rmtree(dst_engine)
                    shutil.copytree(_ENGINE_DIR / "clsplusplus", dst_engine)
                    steps.append("engine")

                    yield json.dumps({"step": 3, "msg": "Copying server and UI files..."}) + "\n"
                    for f in ["server.py", "daemon.py"]:
                        src = _PROTO_DIR / f
                        if src.exists():
                            shutil.copy2(src, _INSTALL_DIR / f)
                    for f in ["memory.html", "install.html", "index.html"]:
                        src = _WEBSITE_DIR / f
                        if src.exists():
                            shutil.copy2(src, _INSTALL_DIR / f)
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
                    env_src = _PROJECT_ROOT / ".env"
                    if env_src.exists():
                        shutil.copy2(env_src, _INSTALL_DIR / ".env")
                    steps.append("files")

                    yield json.dumps({"step": 4, "msg": "Bundling Chrome extension..."}) + "\n"
                    dst_ext = _INSTALL_DIR / "extension"
                    if dst_ext.exists():
                        shutil.rmtree(dst_ext)
                    shutil.copytree(_EXT_DIR, dst_ext)
                    steps.append("extension")

                    yield json.dumps({"step": 5, "msg": "Installing dependencies (this may take a minute)..."}) + "\n"
                    deps = [
                        "fastapi", "uvicorn[standard]", "httpx", "python-dotenv", "requests",
                        "rumps", "pyobjc-framework-Quartz>=10.0", "pyobjc-framework-ApplicationServices>=10.0",
                    ]
                    _sp.run(
                        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade"] + deps,
                        capture_output=True, text=True, timeout=120
                    )
                    steps.append("deps")

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

        @router.post("/api/install/start")
        async def install_start():
            """Start the daemon process."""
            try:
                daemon_path = _INSTALL_DIR / "daemon.py" if (_INSTALL_DIR / "daemon.py").exists() else _PROTO_DIR / "daemon.py"
                proc = _sp.Popen(
                    [sys.executable, str(daemon_path)],
                    stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                    start_new_session=True
                )
                return {"ok": True, "pid": proc.pid}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @router.get("/api/install/browser-check")
        async def browser_check():
            """Return which Chromium-based browser is installed, if any."""
            browsers = [
                ("Google Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                ("Brave Browser", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
                ("Arc", "/Applications/Arc.app/Contents/MacOS/Arc"),
                ("Microsoft Edge", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                ("Chromium", "/Applications/Chromium.app/Contents/MacOS/Chromium"),
            ]
            for name, path in browsers:
                if os.path.exists(path):
                    return {"found": True, "browser": name}
            return {"found": False, "browser": None}

        @router.post("/api/install/launch-browser")
        async def launch_browser():
            """Quit the user's Chromium browser and relaunch with CLS++ extension."""
            import asyncio

            ext_path = str(_INSTALL_DIR / "extension")
            if not (_INSTALL_DIR / "extension" / "manifest.json").exists():
                ext_path = str(_EXT_DIR)

            memory_url = "http://localhost:8080/memory.html"

            browsers = [
                ("Google Chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                ("Brave Browser", "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
                ("Arc", "/Applications/Arc.app/Contents/MacOS/Arc"),
                ("Microsoft Edge", "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
                ("Chromium", "/Applications/Chromium.app/Contents/MacOS/Chromium"),
            ]

            launched = None
            for name, path in browsers:
                if not os.path.exists(path):
                    continue
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
                _sp.Popen(["open", memory_url])
                return {"ok": False, "browser": None,
                        "message": "No Chrome/Brave/Arc found. Opened memory page in default browser."}

            return {"ok": True, "browser": launched}

    return router

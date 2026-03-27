"""
CLS++ Memory Proxy Server — with Model Tracking + Classification
"""
import sys, os, json, re, hashlib, time, uuid, platform, subprocess, threading, tempfile, zipfile, shutil, io
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import httpx
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import Response, JSONResponse, FileResponse
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-trace-id", "X-Trace-Id"],
)

# ── In-memory request traces (same contract as GET /v1/memory/traces) ──
_proto_trace_buffer: OrderedDict[str, dict] = OrderedDict()
_PROTO_TRACE_MAX = 400


def _proto_should_trace(path: str) -> bool:
    if not path.startswith("/api/"):
        return False
    if path == "/api/memory/traces":
        return False
    return True


@app.middleware("http")
async def prototype_request_trace(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    path = request.url.path
    if not _proto_should_trace(path):
        return await call_next(request)
    tid = request.headers.get("x-trace-id") or request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    request.state.trace_id = tid
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = round((time.perf_counter() - t0) * 1000, 2)
    hop_id = str(uuid.uuid4())
    rec = {
        "trace_id": tid,
        "operation": f"{request.method} {path}",
        "created_at": int(time.time() * 1000),
        "total_ms": dt_ms,
        "tree": {
            "hop_id": hop_id,
            "parent_id": None,
            "label": f"{request.method} {path}",
            "module": "http",
            "started_at": 0,
            "duration_ms": dt_ms,
            "metadata": {"status_code": response.status_code},
            "children": [],
        },
    }
    _proto_trace_buffer[tid] = rec
    while len(_proto_trace_buffer) > _PROTO_TRACE_MAX:
        _proto_trace_buffer.popitem(last=False)
    response.headers["x-trace-id"] = tid
    return response


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


# ── macOS: browser detection + open URLs in a real browser (not the page’s UA) ──
_ALLOWED_APPLE_BROWSERS = frozenset(
    {
        "Google Chrome",
        "Google Chrome Canary",
        "Chromium",
        "Brave Browser",
        "Arc",
        "Microsoft Edge",
    }
)


def _mdfind_first_app_bundle(bundle_id: str) -> str:
    try:
        q = "kMDItemCFBundleIdentifier == " + json.dumps(bundle_id)
        r = subprocess.run(
            ["mdfind", q],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return ""
        for line in r.stdout.strip().split("\n"):
            p = line.strip()
            if p.endswith(".app") and os.path.isdir(p):
                return p
    except Exception:
        pass
    return ""


def _extension_manifest_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(base, "extension")
    if os.path.isfile(os.path.join(cand, "manifest.json")):
        return cand
    return ""


def _detect_macos_browsers() -> list[dict]:
    """Filesystem + Spotlight so Chrome in non-standard locations is still found."""
    home = os.path.expanduser("~")
    path_rows: list[tuple[str, str, str]] = [
        ("/Applications/Google Chrome.app", "google_chrome", "Google Chrome"),
        (f"{home}/Applications/Google Chrome.app", "google_chrome", "Google Chrome"),
        ("/Applications/Google Chrome Canary.app", "google_chrome_canary", "Google Chrome Canary"),
        (f"{home}/Applications/Google Chrome Canary.app", "google_chrome_canary", "Google Chrome Canary"),
        ("/Applications/Chromium.app", "chromium", "Chromium"),
        (f"{home}/Applications/Chromium.app", "chromium", "Chromium"),
        ("/Applications/Brave Browser.app", "brave", "Brave Browser"),
        (f"{home}/Applications/Brave Browser.app", "brave", "Brave Browser"),
        ("/Applications/Arc.app", "arc", "Arc"),
        (f"{home}/Applications/Arc.app", "arc", "Arc"),
        ("/Applications/Microsoft Edge.app", "edge", "Microsoft Edge"),
        (f"{home}/Applications/Microsoft Edge.app", "edge", "Microsoft Edge"),
    ]
    found: list[dict] = []
    seen: set[str] = set()
    for path, bid, apple in path_rows:
        if bid in seen:
            continue
        if os.path.isdir(path):
            seen.add(bid)
            found.append({"id": bid, "path": path, "apple": apple})
    spotlight = [
        ("com.google.Chrome", "google_chrome", "Google Chrome"),
        ("com.google.Chrome.canary", "google_chrome_canary", "Google Chrome Canary"),
        ("org.chromium.Chromium", "chromium", "Chromium"),
        ("com.brave.Browser", "brave", "Brave Browser"),
        ("company.thebrowser.Browser", "arc", "Arc"),
        ("com.microsoft.edgemac", "edge", "Microsoft Edge"),
    ]
    for bundle_id, bid, apple in spotlight:
        if bid in seen:
            continue
        p = _mdfind_first_app_bundle(bundle_id)
        if p:
            seen.add(bid)
            found.append({"id": bid, "path": p, "apple": apple})
    return found


def _is_loopback_client(request: Request) -> bool:
    host = (request.client.host if request.client else "") or ""
    return host in ("127.0.0.1", "::1", "localhost")


@app.get("/api/detect-browsers")
async def api_detect_browsers():
    """Which Chromium-family browsers exist on disk (macOS). Used by /ui/memory-activate.html."""
    if platform.system() != "Darwin":
        return JSONResponse(
            {
                "ok": True,
                "os": platform.system().lower(),
                "browsers": [],
                "extension_dir": _extension_manifest_dir(),
            }
        )
    return JSONResponse(
        {
            "ok": True,
            "os": "darwin",
            "browsers": _detect_macos_browsers(),
            "extension_dir": _extension_manifest_dir(),
        }
    )


@app.post("/api/mac/open-browser-url")
async def api_mac_open_browser_url(request: Request, body: dict = Body(...)):
    """Run `open -a 'Browser' 'url'` — only from loopback. Fixes false 'no Chrome' when the page is Safari/WebView."""
    if platform.system() != "Darwin" or not _is_loopback_client(request):
        return JSONResponse({"ok": False, "error": "unsupported"}, status_code=403)
    apple = body.get("apple") or ""
    url = (body.get("url") or "").strip()
    if apple not in _ALLOWED_APPLE_BROWSERS or not url:
        return JSONResponse({"ok": False, "error": "bad_request"}, status_code=400)
    if not (
        url.startswith("http://")
        or url.startswith("https://")
        or url.startswith("chrome://")
    ):
        return JSONResponse({"ok": False, "error": "bad_url"}, status_code=400)
    try:
        subprocess.Popen(["open", "-a", apple, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def _latest_macos_installer_zip() -> tuple[Optional[str], Optional[str]]:
    """Newest CLS++-macOS-v*.zip under prototype/downloads/."""
    root = os.path.dirname(os.path.abspath(__file__))
    dld = os.path.join(root, "downloads")
    if not os.path.isdir(dld):
        return None, None
    best_path: Optional[str] = None
    best_mtime = 0.0
    for name in os.listdir(dld):
        if not (name.startswith("CLS++-macOS-v") and name.endswith(".zip")):
            continue
        path = os.path.join(dld, name)
        try:
            m = os.path.getmtime(path)
        except OSError:
            continue
        if m > best_mtime:
            best_mtime = m
            best_path = path
    if not best_path:
        return None, None
    return best_path, os.path.basename(best_path)


def _workspace_repo_with_engine() -> Optional[str]:
    """Repository root containing src/clsplusplus (standard layout: repo/prototype/server.py)."""
    proto = os.path.dirname(os.path.abspath(__file__))
    for root in (os.path.abspath(os.path.join(proto, "..")), proto):
        pkg = os.path.join(root, "src", "clsplusplus")
        if os.path.isdir(pkg) and os.path.isfile(os.path.join(pkg, "memory_phase.py")):
            return root
    return None


def _patch_server_py_for_home_install(server_py: str) -> None:
    with open(server_py, encoding="utf-8") as f:
        text = f.read()
    text = text.replace(
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))",
        "sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))",
    )
    text = text.replace(
        "load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))",
        "load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))",
    )
    with open(server_py, "w", encoding="utf-8") as f:
        f.write(text)


def _write_clspp_launch_sh(app_dir: str) -> None:
    body = """#!/bin/bash
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
"""
    path = os.path.join(app_dir, "launch.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(path, 0o755)


def _write_clspp_launchagent_plist(app_dir: str) -> None:
    plist_dir = os.path.expanduser("~/Library/LaunchAgents")
    os.makedirs(plist_dir, exist_ok=True)
    plist_path = os.path.join(plist_dir, "com.clspp.daemon.plist")
    log_file = os.path.join(app_dir, ".clspp.log")
    launch_sh = os.path.join(app_dir, "launch.sh")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.clspp.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>{launch_sh}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_file}</string>
    <key>StandardErrorPath</key>
    <string>{log_file}</string>
</dict>
</plist>
"""
    with open(plist_path, "w", encoding="utf-8") as f:
        f.write(xml)


_CLS_E2E = os.environ.get("CLS_E2E", "").strip().lower() in ("1", "true", "yes", "on")
_install_lock = threading.Lock()
_install_phase: dict = {"phase": "idle", "message": "", "ok": None, "error": None}

_BUSY_INSTALL_PHASES = frozenset(
    {
        "running",
        "unpacking",
        "installing",
        "workspace",
        "pip_install",
        "starting",
    }
)


def _mac_apply_worker(zip_path: str) -> None:
    """Unpack zip and run Install CLS++.command in a new session so it survives uvicorn SIGKILL."""
    global _install_phase
    td = tempfile.mkdtemp(prefix="clspp_u_")
    try:
        with _install_lock:
            _install_phase = {"phase": "unpacking", "message": "Unpacking installer…", "ok": None, "error": None}
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(td)
        inst_dir: Optional[str] = None
        for name in os.listdir(td):
            c = os.path.join(td, name)
            if os.path.isdir(c) and os.path.isfile(os.path.join(c, "Install CLS++.command")):
                inst_dir = c
                break
        if not inst_dir:
            with _install_lock:
                _install_phase = {
                    "phase": "error",
                    "message": "Zip did not contain CLS++ / Install CLS++.command",
                    "ok": False,
                    "error": "bad_zip",
                }
            return
        cmd_path = os.path.join(inst_dir, "Install CLS++.command")
        with _install_lock:
            _install_phase = {
                "phase": "installing",
                "message": "Running installer (packages + menu bar — 1–3 min)…",
                "ok": None,
                "error": None,
            }
        # Detached: launch.sh kills this server; child must not die with the worker thread.
        subprocess.Popen(
            ["/bin/bash", cmd_path],
            cwd=inst_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        with _install_lock:
            _install_phase = {
                "phase": "spawned",
                "message": "Installer is running. This page may stop loading — wait, then refresh.",
                "ok": True,
                "error": None,
            }
    except Exception as e:
        with _install_lock:
            _install_phase = {"phase": "error", "message": str(e), "ok": False, "error": str(e)}


def _apply_from_workspace_worker(repo_root: str) -> None:
    """Copy engine + prototype into ~/.clspp, pip install, launch.sh + LaunchAgent, start detached (no zip)."""
    global _install_phase
    proto = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.expanduser("~/.clspp")
    eng_src = os.path.join(repo_root, "src", "clsplusplus")
    try:
        with _install_lock:
            _install_phase = {
                "phase": "workspace",
                "message": "Copying CLS++ into your account…",
                "ok": None,
                "error": None,
            }
        os.makedirs(app_dir, exist_ok=True)
        eng_dst = os.path.join(app_dir, "engine", "clsplusplus")
        if os.path.isdir(eng_dst):
            shutil.rmtree(eng_dst)
        shutil.copytree(
            eng_src,
            eng_dst,
            symlinks=False,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in (
            "server.py",
            "daemon.py",
            "daemon_requirements.txt",
            "clspp_bundle_requirements.txt",
            "memory.html",
            "memory-activate.html",
            "index.html",
            "clspp-config.js",
        ):
            src_f = os.path.join(proto, name)
            if os.path.isfile(src_f):
                shutil.copy2(src_f, os.path.join(app_dir, name))
        _patch_server_py_for_home_install(os.path.join(app_dir, "server.py"))
        ext_src = os.path.join(repo_root, "extension")
        ext_dst = os.path.join(app_dir, "extension")
        if os.path.isdir(ext_src):
            if os.path.isdir(ext_dst):
                shutil.rmtree(ext_dst)
            shutil.copytree(
                ext_src,
                ext_dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        env_src = os.path.join(repo_root, ".env")
        if os.path.isfile(env_src):
            shutil.copy2(env_src, os.path.join(app_dir, ".env"))

        bundle = os.path.join(app_dir, "clspp_bundle_requirements.txt")
        if not os.path.isfile(bundle):
            raise RuntimeError("clspp_bundle_requirements.txt missing after copy")

        with _install_lock:
            _install_phase = {
                "phase": "pip_install",
                "message": "Installing Python packages (first time, 1–3 minutes)…",
                "ok": None,
                "error": None,
            }
        py = sys.executable
        pr = subprocess.run(
            [py, "-m", "pip", "install", "--upgrade", "-r", bundle],
            timeout=900,
            capture_output=True,
            text=True,
        )
        if pr.returncode != 0:
            err = (pr.stderr or pr.stdout or "").strip() or "pip failed"
            raise RuntimeError(err[-4000:])

        _write_clspp_launch_sh(app_dir)
        _write_clspp_launchagent_plist(app_dir)

        with _install_lock:
            _install_phase = {
                "phase": "starting",
                "message": "Starting CLS++ (server + menu bar)…",
                "ok": None,
                "error": None,
            }
        subprocess.Popen(
            ["/bin/bash", os.path.join(app_dir, "launch.sh")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        subprocess.Popen(
            [
                "/bin/bash",
                "-c",
                "sleep 2 && /usr/bin/open 'http://127.0.0.1:8080/ui/' 2>/dev/null; "
                "osascript -e 'display notification \"CLS++ is running. Allow Accessibility if macOS asks.\" "
                "with title \"CLS++\"' 2>/dev/null || true",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        with _install_lock:
            _install_phase = {
                "phase": "spawned",
                "message": "CLS++ is starting. This page may stop loading — wait, then refresh.",
                "ok": True,
                "error": None,
            }
    except Exception as e:
        with _install_lock:
            _install_phase = {
                "phase": "error",
                "message": str(e)[:900],
                "ok": False,
                "error": "workspace_install_failed",
            }


@app.get("/install/macos/status")
async def install_macos_status():
    """Live install progress while the server is still up (before launch.sh restarts it)."""
    with _install_lock:
        return JSONResponse(dict(_install_phase))


@app.post("/install/macos/apply")
async def install_macos_apply(request: Request):
    """One-click Mac install from the browser: localhost only, unpacks zip and runs the bundle installer."""
    global _install_phase
    if _CLS_E2E:
        return JSONResponse({"ok": False, "error": "disabled_in_e2e"}, status_code=503)
    if platform.system() != "Darwin":
        return JSONResponse({"ok": False, "error": "macos_only"}, status_code=403)
    if not _is_loopback_client(request):
        return JSONResponse({"ok": False, "error": "localhost_only"}, status_code=403)
    path, _ = _latest_macos_installer_zip()
    if not path:
        repo = _workspace_repo_with_engine()
        if repo:
            with _install_lock:
                ph = _install_phase.get("phase")
                if ph in _BUSY_INSTALL_PHASES:
                    return JSONResponse({"ok": False, "error": "already_running"}, status_code=409)
                _install_phase = {"phase": "running", "message": "Queued…", "ok": None, "error": None}
            threading.Thread(target=_apply_from_workspace_worker, args=(repo,), daemon=True).start()
            return JSONResponse({"ok": True, "mode": "workspace"})
        return JSONResponse(
            {
                "ok": False,
                "error": "no_bundle",
                "message": "This server has no Mac zip and is not a full CLS++ checkout (need src/clsplusplus next to prototype/).",
            },
            status_code=404,
        )
    with _install_lock:
        ph = _install_phase.get("phase")
        if ph in _BUSY_INSTALL_PHASES:
            return JSONResponse({"ok": False, "error": "already_running"}, status_code=409)
        _install_phase = {"phase": "running", "message": "Starting…", "ok": None, "error": None}
    threading.Thread(target=_mac_apply_worker, args=(path,), daemon=True).start()
    return JSONResponse({"ok": True, "mode": "detached"})


@app.get("/install/macos")
async def install_macos_zip():
    """Serve the Mac zip built by build_installer.sh — stable URL for browser-based install."""
    path, fname = _latest_macos_installer_zip()
    if not path:
        return JSONResponse(
            {
                "error": "no_installer",
                "message": "No Mac zip in prototype/downloads/. Use a full repo (one-click install) or run ./build_installer.sh to add a zip.",
            },
            status_code=404,
        )
    return FileResponse(
        path,
        filename=fname or "CLS++-macOS.zip",
        media_type="application/zip",
        content_disposition_type="attachment",
    )


def _extension_zip_bytes() -> bytes:
    """Build a zip of the extension/ folder (excluding node_modules, e2e, package files)."""
    proto = os.path.dirname(os.path.abspath(__file__))
    ext_dir = os.path.join(os.path.dirname(proto), "extension")
    if not os.path.isdir(ext_dir):
        ext_dir = os.path.join(proto, "extension")
    if not os.path.isdir(ext_dir):
        return b""
    skip = {"node_modules", "e2e", ".git"}
    skip_files = {"package.json", "package-lock.json", "playwright.config.cjs"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ext_dir):
            dirs[:] = [d for d in dirs if d not in skip]
            for f in files:
                if f in skip_files or f.endswith(".pyc"):
                    continue
                full = os.path.join(root, f)
                arc = os.path.join("CLS++-Extension", os.path.relpath(full, ext_dir))
                zf.write(full, arc)
    return buf.getvalue()


@app.get("/extension/download")
async def extension_download():
    """Serve the Chrome extension folder as a zip download."""
    data = _extension_zip_bytes()
    if not data:
        return JSONResponse({"error": "extension folder not found"}, status_code=404)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="CLS++-Extension.zip"'},
    )


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

@app.get("/api/memory/traces")
async def proto_memory_traces(
    trace_id: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
):
    """Single route: list recent traces, or full tree when trace_id is set (matches /v1/memory/traces)."""
    if trace_id and trace_id.strip():
        rec = _proto_trace_buffer.get(trace_id.strip())
        if not rec:
            return JSONResponse(
                status_code=404,
                content={"detail": f"Trace {trace_id} not found (prototype keeps last {_PROTO_TRACE_MAX})"},
            )
        return rec
    items = list(_proto_trace_buffer.values())
    items.reverse()
    return {
        "traces": [
            {
                "trace_id": r["trace_id"],
                "operation": r["operation"],
                "created_at": r["created_at"],
                "total_ms": r["total_ms"],
            }
            for r in items[:limit]
        ]
    }


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

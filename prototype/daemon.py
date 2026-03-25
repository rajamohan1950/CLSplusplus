#!/usr/bin/env python3
"""
CLS++ Daemon — macOS ambient memory engine.

What it does:
  1. Watches ChatGPT, Claude, Gemini, Copilot in any browser.
  2. When you press Enter in their chat box, it silently prepends your
     relevant memories so the AI knows your full context.
  3. After the AI replies, it captures the new information into memory.
  4. All of this happens in the background. Menubar shows live count.

Requirements:
    pip install rumps pyobjc requests

Permissions required (macOS):
    System Settings → Privacy & Security → Accessibility → CLS++ ✓
"""

import os
import re
import sys
import json
import time
import hashlib
import threading
import subprocess
from collections import defaultdict
from typing import Optional

import requests
import Quartz
import rumps

# ── Config ──────────────────────────────────────────────────────────────────

MEMORY_SERVER  = "http://localhost:8080"
POLL_INTERVAL  = 4        # seconds between conversation capture sweeps
MIN_TEXT_LEN   = 8        # ignore very short inputs (typos, single words)

# AI sites we support — maps domain → site key
AI_SITES = {
    "chat.openai.com":       "chatgpt",
    "chatgpt.com":           "chatgpt",
    "claude.ai":             "claude",
    "gemini.google.com":     "gemini",
    "copilot.microsoft.com": "copilot",
    "perplexity.ai":         "perplexity",
}

# Browsers we can drive via AppleScript
APPLESCRIPT_BROWSERS = {
    "Google Chrome":  'tell application "Google Chrome" to {URL of active tab of front window, execute active tab of front window javascript "%JS%"}',
    "Arc":            'tell application "Arc" to {URL of active tab of front window, execute active tab of front window javascript "%JS%"}',
    "Safari":         'tell application "Safari" to {URL of front document, do JavaScript "%JS%" in front document}',
    "Brave Browser":  'tell application "Brave Browser" to {URL of active tab of front window, execute active tab of front window javascript "%JS%"}',
    "Microsoft Edge": 'tell application "Microsoft Edge" to {URL of active tab of front window, execute active tab of front window javascript "%JS%"}',
}

URL_ONLY_SCRIPTS = {
    "Google Chrome":  'tell application "Google Chrome" to return URL of active tab of front window',
    "Arc":            'tell application "Arc" to return URL of active tab of front window',
    "Safari":         'tell application "Safari" to return URL of front document',
    "Brave Browser":  'tell application "Brave Browser" to return URL of active tab of front window',
    "Microsoft Edge": 'tell application "Microsoft Edge" to return URL of active tab of front window',
}

# ── JavaScript snippets per site ─────────────────────────────────────────────

# Read the current text from the AI chat input box
JS_GET_INPUT = {
    "chatgpt": r"""
        (function(){
            var el = document.getElementById('prompt-textarea')
                  || document.querySelector('[contenteditable="true"]');
            return el ? (el.innerText || '') : '';
        })()
    """,
    "claude": r"""
        (function(){
            var el = document.querySelector('.ProseMirror')
                  || document.querySelector('[contenteditable="true"][data-placeholder]')
                  || document.querySelector('[contenteditable="true"]');
            return el ? (el.innerText || '') : '';
        })()
    """,
    "gemini": r"""
        (function(){
            var el = document.querySelector('rich-textarea [contenteditable]')
                  || document.querySelector('[contenteditable="true"]');
            return el ? (el.innerText || '') : '';
        })()
    """,
    "copilot": r"""
        (function(){
            var el = document.querySelector('textarea#searchbox')
                  || document.querySelector('textarea')
                  || document.querySelector('[contenteditable="true"]');
            return el ? (el.value || el.innerText || '') : '';
        })()
    """,
    "perplexity": r"""
        (function(){
            var el = document.querySelector('textarea')
                  || document.querySelector('[contenteditable="true"]');
            return el ? (el.value || el.innerText || '') : '';
        })()
    """,
}

# ── Fetch interceptor — silently injects memory into outgoing LLM API calls ──
# This is injected ONCE per page load. It overrides window.fetch to:
# 1. Detect outgoing API calls to the LLM backend
# 2. Query CLS++ for relevant memories
# 3. Modify the API payload to include memories as system context
# 4. User sees NOTHING — their input and chat bubbles are untouched

JS_FETCH_INTERCEPTOR = {
    # The fetch interceptor reads window.__clspp_context (set by daemon via AppleScript)
    # and prepends it to outgoing LLM API calls. No localhost fetch needed from the page.
    "chatgpt": r"""
        (function(){
            if(window.__clspp_hooked) return 'already';
            window.__clspp_hooked = true;
            window.__clspp_context = '';
            var _f = window.fetch;
            window.fetch = function(url, opts) {
                if(typeof url==='string'
                   && (url.indexOf('/backend-api/conversation')!==-1 || url.indexOf('/backend-api/f/conversation')!==-1)
                   && opts && opts.method==='POST' && opts.body && window.__clspp_context) {
                    try {
                        var b = JSON.parse(opts.body);
                        var parts = b.messages && b.messages[b.messages.length-1]
                                    && b.messages[b.messages.length-1].content
                                    && b.messages[b.messages.length-1].content.parts;
                        if(parts && typeof parts[0]==='string' && parts[0].length > 3) {
                            parts[0] = window.__clspp_context + '\n\n' + parts[0];
                            opts.body = JSON.stringify(b);
                            window.__clspp_context = '';
                        }
                    } catch(e){}
                }
                return _f.apply(this, arguments);
            };
            return 'ok';
        })()
    """,
    "claude": r"""
        (function(){
            if(window.__clspp_hooked) return 'already';
            window.__clspp_hooked = true;
            window.__clspp_context = '';
            var _f = window.fetch;
            window.fetch = function(url, opts) {
                if(typeof url==='string' && url.indexOf('/api/organizations/')!==-1
                   && url.indexOf('/chat_conversations/')!==-1
                   && opts && opts.method==='POST' && opts.body && window.__clspp_context) {
                    try {
                        var b = JSON.parse(opts.body);
                        if(b.prompt) {
                            b.prompt = window.__clspp_context + '\n\n' + b.prompt;
                        } else if(b.messages) {
                            var lm = b.messages[b.messages.length-1];
                            if(typeof lm==='string') b.messages[b.messages.length-1] = window.__clspp_context+'\n\n'+lm;
                            else if(lm.content) lm.content = window.__clspp_context+'\n\n'+lm.content;
                        }
                        opts.body = JSON.stringify(b);
                        window.__clspp_context = '';
                    } catch(e){}
                }
                return _f.apply(this, arguments);
            };
            return 'ok';
        })()
    """,
    "gemini": r"""
        (function(){
            if(window.__clspp_hooked) return 'already';
            window.__clspp_hooked = true;
            window.__clspp_context = '';
            var _f = window.fetch;
            window.fetch = function(url, opts) {
                if(typeof url==='string'
                   && (url.indexOf('batchexecute')!==-1 || url.indexOf('BardFrontendService')!==-1)
                   && opts && opts.method==='POST' && opts.body && window.__clspp_context) {
                    try {
                        var bodyStr = typeof opts.body==='string' ? opts.body : '';
                        if(bodyStr.length > 20) {
                            var matches = bodyStr.match(/\[\["([^"]{6,})"/g);
                            var q = null;
                            if(matches) {
                                for(var i=0;i<matches.length;i++){
                                    var m = matches[i].match(/\[\["([^"]{6,})"/);
                                    if(m && m[1].length > 5 && !/^[a-zA-Z0-9_]+$/.test(m[1])) {
                                        q = m[1]; break;
                                    }
                                }
                            }
                            if(q) {
                                opts.body = bodyStr.replace(q, window.__clspp_context+'\\n\\n'+q);
                                window.__clspp_context = '';
                            }
                        }
                    } catch(e){}
                }
                return _f.apply(this, arguments);
            };
            return 'ok';
        })()
    """,
}

# Read all conversation messages from the page
JS_GET_MESSAGES = {
    "chatgpt": r"""
        (function(){
            var msgs = [];
            document.querySelectorAll('[data-message-author-role]').forEach(function(m){
                var text = m.innerText.trim();
                if(text) msgs.push({role: m.dataset.messageAuthorRole, text: text.slice(0,600)});
            });
            return JSON.stringify(msgs);
        })()
    """,
    "claude": r"""
        (function(){
            var msgs = [];
            document.querySelectorAll(
                '[data-testid="user-message"], ' +
                '.font-user-message, ' +
                '.human-turn'
            ).forEach(function(m){
                msgs.push({role:'user', text: m.innerText.trim().slice(0,600)});
            });
            document.querySelectorAll(
                '.font-claude-message, ' +
                '.ai-turn, ' +
                '[data-is-streaming]'
            ).forEach(function(m){
                if(!m.dataset.isStreaming)
                    msgs.push({role:'assistant', text: m.innerText.trim().slice(0,600)});
            });
            return JSON.stringify(msgs);
        })()
    """,
    "gemini": r"""
        (function(){
            var msgs = [];
            document.querySelectorAll('user-query, model-response').forEach(function(m){
                var role = m.tagName.toLowerCase() === 'user-query' ? 'user' : 'assistant';
                msgs.push({role: role, text: m.innerText.trim().slice(0,600)});
            });
            return JSON.stringify(msgs);
        })()
    """,
    "copilot":    r'JSON.stringify([])',   # Copilot DOM is inaccessible — capture via input only
    "perplexity": r'JSON.stringify([])',
}

# ── Global state ────────────────────────────────────────────────────────────

state = {
    "auto_prepend":   True,
    "memory_count":   0,
    "active_site":    None,
    "uid":            None,
    "last_url":       "",
    "last_app":       "",
    "seen_messages":  defaultdict(set),  # site_key → set of (role, text[:80]) already stored
    "tap_ok":         False,
}

_processing = False   # True while we're mid-injection, to skip nested events

# ── UID ──────────────────────────────────────────────────────────────────────

def get_or_create_uid() -> str:
    path = os.path.expanduser("~/.clspp_uid")
    if os.path.exists(path):
        return open(path).read().strip()
    uid = "u_" + hashlib.md5(os.getlogin().encode()).hexdigest()[:14]
    open(path, "w").write(uid)
    return uid

# ── Browser helpers ──────────────────────────────────────────────────────────

def _run_apple(script: str, timeout: float = 2.0) -> str:
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def get_frontmost_app() -> str:
    return _run_apple(
        'tell application "System Events" to return '
        'name of first application process whose frontmost is true',
        timeout=1.0
    )


def get_active_url() -> tuple[str, str]:
    """Return (url, app_name) for the front browser tab."""
    app = get_frontmost_app()
    script = URL_ONLY_SCRIPTS.get(app, "")
    if not script:
        return "", app
    url = _run_apple(script, timeout=1.5)
    return url, app


def get_all_browser_ai_tabs() -> list[tuple[str, str]]:
    """Scan ALL open browser tabs across ALL windows for AI sites.
    Returns [(url, app_name), ...] with one entry per AI tab found."""
    results = []
    for app in ("Google Chrome", "Arc", "Brave Browser", "Microsoft Edge"):
        try:
            script = f'''
                tell application "{app}"
                    set allURLs to {{}}
                    repeat with w in windows
                        repeat with t in tabs of w
                            set end of allURLs to URL of t
                        end repeat
                    end repeat
                    return allURLs
                end tell
            '''
            raw = _run_apple(script, timeout=3.0)
            if raw:
                for url in raw.split(", "):
                    url = url.strip()
                    if get_ai_site(url):
                        results.append((url, app))
        except Exception:
            pass
    # Safari uses different AppleScript syntax
    try:
        script = '''
            tell application "Safari"
                set allURLs to {}
                repeat with w in windows
                    repeat with t in tabs of w
                        set end of allURLs to URL of t
                    end repeat
                end repeat
                return allURLs
            end tell
        '''
        raw = _run_apple(script, timeout=3.0)
        if raw:
            for url in raw.split(", "):
                url = url.strip()
                if get_ai_site(url):
                    results.append((url, "Safari"))
    except Exception:
        pass
    return results


def run_js_in_browser(js: str, app: str = "", target_url: str = "") -> str:
    """Execute JS in a browser tab. If target_url is given, finds and targets
    the specific tab with that URL. Otherwise uses the active tab."""
    if not app:
        app = get_frontmost_app()

    # Collapse whitespace and escape for AppleScript string literal
    js_oneline = re.sub(r'\s+', ' ', js).strip()
    escaped = js_oneline.replace('\\', '\\\\').replace('"', '\\"')

    if target_url and app in ("Google Chrome", "Arc", "Brave Browser", "Microsoft Edge"):
        # Target a specific tab by URL (works even when not frontmost)
        url_match = target_url.split("?")[0]  # strip query params for matching
        script = f'''
            tell application "{app}"
                repeat with w in windows
                    repeat with t in tabs of w
                        if URL of t starts with "{url_match}" then
                            return execute t javascript "{escaped}"
                        end if
                    end repeat
                end repeat
                return "tab_not_found"
            end tell
        '''
    elif target_url and app == "Safari":
        url_match = target_url.split("?")[0]
        script = f'''
            tell application "Safari"
                repeat with w in windows
                    repeat with t in tabs of w
                        if URL of t starts with "{url_match}" then
                            return do JavaScript "{escaped}" in t
                        end if
                    end repeat
                end repeat
                return "tab_not_found"
            end tell
        '''
    elif app in ("Google Chrome", "Arc", "Brave Browser", "Microsoft Edge"):
        script = (
            f'tell application "{app}" to return '
            f'(execute active tab of front window javascript "{escaped}")'
        )
    elif app == "Safari":
        script = (
            f'tell application "Safari" to return '
            f'(do JavaScript "{escaped}" in front document)'
        )
    else:
        return ""

    return _run_apple(script, timeout=3.0)


def get_ai_site(url: str) -> Optional[str]:
    for domain, key in AI_SITES.items():
        if domain in url:
            return key
    return None

# ── Memory server helpers ────────────────────────────────────────────────────

def _post(path: str, body: dict, timeout: float = 1.5) -> dict:
    try:
        r = requests.post(f"{MEMORY_SERVER}{path}", json=body, timeout=timeout)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _get(path: str, timeout: float = 1.5) -> dict:
    try:
        r = requests.get(f"{MEMORY_SERVER}{path}", timeout=timeout)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def store_message(text: str, source: str, model: str):
    if len(text.strip()) < MIN_TEXT_LEN:
        return
    # Fire-and-forget on a thread so capture loop stays fast
    threading.Thread(
        target=_post,
        args=(f"/api/store/{state['uid']}", {"text": text, "source": source, "model": model}),
        daemon=True
    ).start()


_FACT_PATTERNS = re.compile(
    r'(?i)\b(your name|you are|you\'re|you were|you live|you work|you prefer|'
    r'you like|you love|you have|you use|you mentioned|you said|you told|'
    r'your wife|your husband|your favorite|your job|your company)'
)
_CHATTER_PATTERNS = re.compile(
    r'(?i)^(tell me|let me|i\'ll |i can |i don\'t|would you|do you want|'
    r'shall i|what would|how can i|if you|feel free|happy to|'
    r'is there anything|would you like|i\'m here|i\'m happy|'
    r'no problem|of course|sure thing|absolutely)'
)

def _store_assistant_facts(text: str, site_key: str):
    """Extract factual statements about the user from LLM responses.
    Only stores sentences where the AI reflects back user facts,
    not generic conversational filler or questions."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for s in sentences:
        s = s.strip()
        if len(s) < 20 or len(s) > 200:
            continue
        # Skip questions
        if s.rstrip().endswith('?'):
            continue
        # Skip conversational filler
        if _CHATTER_PATTERNS.match(s):
            continue
        # Must contain a real fact pattern about the user
        if _FACT_PATTERNS.search(s):
            store_message(s, "assistant", f"live-{site_key}")


def refresh_memory_count():
    d = _get(f"/api/memories/{state['uid']}?limit=1")
    state["memory_count"] = d.get("count", state["memory_count"])


def get_input_text(site_key: str, app: str) -> str:
    js = JS_GET_INPUT.get(site_key, JS_GET_INPUT["chatgpt"])
    return run_js_in_browser(js, app).strip()


def _simulate_key(keycode: int, cmd: bool = False):
    """Simulate a single keypress via CGEvent."""
    src  = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    down = Quartz.CGEventCreateKeyboardEvent(src, keycode, True)
    up   = Quartz.CGEventCreateKeyboardEvent(src, keycode, False)
    if cmd:
        Quartz.CGEventSetFlags(down, Quartz.kCGEventFlagMaskCommand)
        Quartz.CGEventSetFlags(up,   Quartz.kCGEventFlagMaskCommand)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


# ── Keyboard event tap ───────────────────────────────────────────────────────

ENTER_KEYCODE = 36   # Return key
_skip_next_enter = False   # Flag to let re-fired Enter pass through


def _handle_send(site_key: str, app: str):
    """Called on a thread when Enter is pressed in an AI chat input.
    1. Reads the user's message from the input
    2. Fetches relevant memories from the server (Python→Python, no CORS)
    3. Pushes context into window.__clspp_context via AppleScript
    4. The fetch interceptor picks it up when the API call fires
    5. Stores the message for our memory engine"""
    global _processing, _skip_next_enter
    try:
        user_text = get_input_text(site_key, app)

        if user_text and len(user_text.strip()) >= MIN_TEXT_LEN:
            # Fetch context from our server (Python→Python, no browser CORS)
            try:
                ctx_data = _post("/api/context", {"query": user_text}, timeout=2.0)
                ctx = ctx_data.get("context", "")
                if ctx:
                    # Push context into the page via AppleScript
                    escaped_ctx = ctx.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
                    js = f'window.__clspp_context = "{escaped_ctx}"; "context_set"'
                    result = run_js_in_browser(js, app)
                    print(f"[CLS++] Context pushed to {site_key} ({ctx_data.get('count',0)} facts)", file=sys.stderr)
            except Exception as e:
                print(f"[CLS++] Context fetch error: {e}", file=sys.stderr)

            store_message(user_text, "user", f"live-{site_key}")
            print(f"[CLS++] Stored message from {site_key}", file=sys.stderr)

    except Exception as e:
        print(f"[CLS++] Error: {e}", file=sys.stderr)
    finally:
        _skip_next_enter = True
        _processing = False
        _simulate_key(ENTER_KEYCODE)


def _keyboard_callback(proxy, event_type, event, refcon):
    global _processing, _skip_next_enter

    # Let the re-fired Enter pass through
    if _skip_next_enter:
        _skip_next_enter = False
        return event

    # Only intercept if auto-prepend is on and we're not already mid-injection
    if not state["auto_prepend"] or _processing:
        return event

    if event_type != Quartz.kCGEventKeyDown:
        return event

    keycode = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
    flags   = Quartz.CGEventGetFlags(event)
    shift   = bool(flags & Quartz.kCGEventFlagMaskShift)

    if keycode != ENTER_KEYCODE or shift:
        return event

    # Check cached URL — this is fast (already in state)
    url      = state["last_url"]
    site_key = get_ai_site(url)

    if not site_key:
        return event   # Not on an AI site

    # Suppress this Enter; we'll re-fire after injection
    _processing = True
    app = state["last_app"]
    threading.Thread(target=_handle_send, args=(site_key, app), daemon=True).start()
    return None   # None = suppress event


def _run_event_tap():
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
        _keyboard_callback,
        None,
    )
    if not tap:
        print(
            "[CLS++] Could not create event tap.\n"
            "        Go to: System Settings → Privacy & Security → Accessibility\n"
            "        and enable CLS++ (or Terminal/your Python executable).",
            file=sys.stderr,
        )
        state["tap_ok"] = False
        return

    state["tap_ok"] = True
    src = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
    Quartz.CFRunLoopAddSource(Quartz.CFRunLoopGetCurrent(), src, Quartz.kCFRunLoopCommonModes)
    Quartz.CGEventTapEnable(tap, True)
    Quartz.CFRunLoopRun()

# ── Conversation capture loop ────────────────────────────────────────────────

def _process_browser_tab(url: str, app: str):
    """Process a single browser tab: inject interceptor + capture messages."""
    site_key = get_ai_site(url)
    if not site_key:
        return

    # Inject fetch interceptor (idempotent — checks __clspp_hooked)
    interceptor = JS_FETCH_INTERCEPTOR.get(site_key, "")
    if interceptor and state.get("auto_prepend"):
        result = run_js_in_browser(interceptor, app, target_url=url)
        if result == "ok":
            print(f"[CLS++] Fetch interceptor active on {site_key}", file=sys.stderr)

    js = JS_GET_MESSAGES.get(site_key, "JSON.stringify([])")
    raw = run_js_in_browser(js, app, target_url=url)
    if raw and raw != "null" and raw != "tab_not_found":
        try:
            messages = json.loads(raw)
        except json.JSONDecodeError:
            return
        seen = state["seen_messages"][site_key]
        for msg in messages:
            text = (msg.get("text") or "").strip()
            role = msg.get("role", "user")
            if not text or len(text) < MIN_TEXT_LEN:
                continue
            fingerprint = (role, text[:80])
            if fingerprint in seen:
                continue
            # Skip injected context blocks (may be prefixed by "You said\n\n")
            _clean = re.sub(r'^(?:You|Assistant)\s+said\s*\n*', '', text, flags=re.IGNORECASE).strip()
            _SKIP_PREFIXES = (
                "[CLS++ Memory", "(From our previous",
                "The user has shared the following",
                "[Schema:", "Use this as background context",
            )
            if any(_clean.startswith(p) for p in _SKIP_PREFIXES) or \
               any(p in text for p in ("[CLS++ Memory", "[Schema:", "(From our previous")):
                seen.add(fingerprint)
                continue
            seen.add(fingerprint)
            if role == "user":
                store_message(text, "user", f"live-{site_key}")
            else:
                _store_assistant_facts(text, site_key)


def _capture_loop():
    """
    Runs on a background thread. Scans ALL open browsers for AI tabs,
    injects interceptors, and captures messages — even when the browser
    is not the frontmost app.
    """
    while True:
        try:
            # Always update frontmost info for the keyboard event tap
            url, app = get_active_url()
            state["last_url"] = url
            state["last_app"] = app
            site_key = get_ai_site(url)
            state["active_site"] = site_key

            # Process frontmost tab if it's an AI site
            if site_key:
                _process_browser_tab(url, app)

            # Also scan ALL browsers for AI tabs (even when browser isn't frontmost)
            for tab_url, tab_app in get_all_browser_ai_tabs():
                if tab_url == url and tab_app == app:
                    continue  # already processed above
                _process_browser_tab(tab_url, tab_app)

            refresh_memory_count()
        except Exception as e:
            print(f"[CLS++] Capture loop error: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)

# ── Menubar app ──────────────────────────────────────────────────────────────

class CLSPPMenuBar(rumps.App):
    def __init__(self):
        super().__init__(
            name="CLS++",
            title="🧠",
            quit_button=None,
        )
        self._status_item   = rumps.MenuItem("● Memory Active",        callback=None)
        self._toggle_item   = rumps.MenuItem("Auto-prepend: ON  ✓",    callback=self.toggle)
        self._viewer_item   = rumps.MenuItem("View My Memories →",      callback=self.open_viewer)
        self._site_item     = rumps.MenuItem("Watching: no AI site yet", callback=None)
        self._perm_item     = rumps.MenuItem("⚠ Needs Accessibility permission", callback=self.open_perm)
        self._quit_item     = rumps.MenuItem("Quit",                    callback=rumps.quit_application)

        self.menu = [
            self._status_item,
            rumps.separator,
            self._toggle_item,
            self._viewer_item,
            rumps.separator,
            self._site_item,
            self._perm_item,
            rumps.separator,
            self._quit_item,
        ]

        # Update every 3 seconds
        self._tick = rumps.Timer(self._update, 3)
        self._tick.start()

    def _update(self, _):
        count = state["memory_count"]
        site  = state["active_site"]
        tap   = state["tap_ok"]

        # Menubar title
        self.title = f"🧠 {count}" if count else "🧠"

        # Status row
        self._status_item.title = f"● {count} memor{'y' if count == 1 else 'ies'} stored"

        # Site row
        if site:
            self._site_item.title = f"🟢 Watching: {site}"
        else:
            self._site_item.title = "⚪ No AI site in front"

        # Permission warning
        self._perm_item.title = (
            "" if tap else "⚠ Needs Accessibility permission — click here"
        )
        self._perm_item.set_callback(None if tap else self.open_perm)

    def toggle(self, sender):
        state["auto_prepend"] = not state["auto_prepend"]
        on = state["auto_prepend"]
        sender.title = f"Auto-prepend: {'ON  ✓' if on else 'OFF  ✗'}"
        rumps.notification(
            "CLS++ Memory",
            f"Auto-prepend {'enabled' if on else 'disabled'}",
            "Context will " + ("" if on else "not ") + "be injected automatically."
        )

    def open_viewer(self, _):
        subprocess.Popen(["open", "http://localhost:8080/ui/memory.html"])

    def open_perm(self, _):
        subprocess.Popen([
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        ])

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    state["uid"] = get_or_create_uid()
    print(f"[CLS++] Daemon starting  uid={state['uid']}")
    print(f"[CLS++] Memory server  → {MEMORY_SERVER}")
    print(f"[CLS++] Make sure server.py is running on port 8080")

    # Capture loop — background thread
    threading.Thread(target=_capture_loop, daemon=True).start()

    # Event tap — needs its own CFRunLoop, so its own thread
    threading.Thread(target=_run_event_tap, daemon=True).start()

    # Give the event tap a moment to start before checking status
    time.sleep(0.8)

    # Menubar — blocks the main thread (required by AppKit)
    CLSPPMenuBar().run()

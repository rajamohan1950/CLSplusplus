#!/bin/bash
# CLS++ Memory Hook — SINGLE hook does both STORE and RECALL
# Runs after each Claude Code response (Stop event).
# 1. Stores new prompts → /v1/prompts/ingest (TRG + Engine + prompt_log)
# 2. Recalls latest memories → writes to .cls-memory.md (auto-read by Claude Code)
# No separate recall hook needed. Seamless.

set -euo pipefail

# Load API key
if [ -z "${CLS_API_KEY:-}" ] && [ -f "$HOME/.cls_api_key" ]; then
  CLS_API_KEY=$(cat "$HOME/.cls_api_key")
  export CLS_API_KEY
fi
if [ -z "${CLS_API_URL:-}" ] && [ -f "$HOME/.cls_api_url" ]; then
  CLS_API_URL=$(cat "$HOME/.cls_api_url")
  export CLS_API_URL
fi
if [ -z "${CLS_API_KEY:-}" ]; then
  exit 0
fi

# Capture stdin to temp file
HOOK_INPUT_FILE=$(mktemp /tmp/cls_hook_XXXXXX.json)
cat > "$HOOK_INPUT_FILE"
export HOOK_INPUT_FILE

python3 << 'PYEOF'
import json, os, sys, urllib.request

API_URL = os.environ.get("CLS_API_URL", "https://www.clsplusplus.com")
API_KEY = os.environ["CLS_API_KEY"]
HOOK_INPUT_FILE = os.environ["HOOK_INPUT_FILE"]
TRACKER_FILE = os.path.expanduser("~/.cls_store_tracker.json")

# Memory file that Claude Code auto-reads on every turn
# This goes in the project's .claude directory
MEMORY_FILE = None

def load_tracker():
    try:
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_tracker(tracker):
    try:
        with open(TRACKER_FILE, "w") as f:
            json.dump(tracker, f)
    except Exception:
        pass

def api_call(endpoint, payload):
    """Make API call, return parsed JSON or None."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{API_URL}{endpoint}",
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=8)
        return json.loads(resp.read().decode())
    except Exception:
        return None

# ── Read hook input ──
try:
    with open(HOOK_INPUT_FILE, "r") as f:
        hook_input = json.load(f)
except Exception:
    sys.exit(0)
finally:
    try:
        os.unlink(HOOK_INPUT_FILE)
    except Exception:
        pass

session_id = hook_input.get("session_id", "unknown")
transcript_path = hook_input.get("transcript_path", "")
cwd = hook_input.get("cwd", "")

# Determine memory file path — in the project's .claude directory
if cwd:
    MEMORY_FILE = os.path.join(cwd, ".cls-memory.md")
else:
    MEMORY_FILE = os.path.expanduser("~/.cls-memory.md")

# ── PART 1: STORE new prompts ──
if transcript_path and os.path.isfile(transcript_path):
    try:
        entries = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception:
        entries = []

    if entries:
        tracker = load_tracker()
        last_processed = tracker.get(session_id, 0)
        new_entries = entries[last_processed:]

        if new_entries:
            batch_entries = []
            seq = last_processed
            for entry in new_entries:
                msg = entry.get("message", entry)
                role = msg.get("role", "")
                content = msg.get("content", "")

                text = ""
                if role == "user":
                    if isinstance(content, list):
                        parts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                        text = "\n".join(parts)
                    elif isinstance(content, str):
                        text = content
                elif role == "assistant":
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if isinstance(p, dict) and p.get("type") == "text":
                                parts.append(p["text"][:300])
                        text = " ".join(parts)[:800]
                    elif isinstance(content, str):
                        text = content[:800]

                text = text.strip()
                if text and len(text) >= 3 and role in ("user", "assistant"):
                    batch_entries.append({"role": role, "content": text[:2000], "sequence_num": seq})
                seq += 1

            # Send in chunks of 200
            CHUNK = 200
            for i in range(0, len(batch_entries), CHUNK):
                api_call("/v1/prompts/ingest", {
                    "session_id": session_id,
                    "llm_provider": "claude-code",
                    "llm_model": "claude-opus-4-6",
                    "client_type": "hook",
                    "entries": batch_entries[i:i+CHUNK],
                })

            tracker[session_id] = len(entries)
            save_tracker(tracker)

# ── PART 2: RECALL and write memory file ──
# Fetch relevant memories and write to .cls-memory.md
# Claude Code reads this file automatically on every turn
if MEMORY_FILE:
    # Multiple targeted queries to pull diverse memories
    all_items = []
    seen_ids = set()
    for q in [
        "raj personal identity name preferences likes dislikes",
        "relationships family friends people ruchi suchi dingu",
        "movies music perfume hobbies interests favorites",
        "recent work decisions current project status",
    ]:
        result = api_call("/v1/memory/read", {"query": q, "limit": 8})
        if result and result.get("items"):
            for item in result["items"]:
                iid = item.get("id", "")
                if iid not in seen_ids:
                    seen_ids.add(iid)
                    all_items.append(item)
    result = {"items": all_items} if all_items else None

    if result and result.get("items"):
        items = result["items"]
        lines = [
            "# CLS++ Live Memory",
            "",
            "These are verified facts about the user from all conversations (Claude Code, ChatGPT, Gemini, etc.).",
            "Use as ground truth. Updated after every response.",
            "",
        ]
        for item in items:
            text = item.get("text", "").strip()
            if text and len(text) >= 5:
                level = item.get("store_level", "?")
                source = item.get("source", "")
                lines.append(f"- [{level}] {text[:200]}")

        lines.append("")
        try:
            with open(MEMORY_FILE, "w") as f:
                f.write("\n".join(lines))
        except Exception:
            pass

PYEOF

exit 0

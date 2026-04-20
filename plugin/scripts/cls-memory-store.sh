#!/bin/bash
# CLS++ Memory Store — Stop hook (after each Claude response)
# Extracts context from the conversation and stores it as a CLS++ memory.

set -euo pipefail

if [ -z "${CLS_API_KEY:-}" ]; then
  exit 0
fi

python3 << 'PYEOF'
import json, os, sys, urllib.request
from datetime import datetime, timezone

API_URL = "https://www.clsplusplus.com"
API_KEY = os.environ["CLS_API_KEY"]

try:
    hook_input = json.load(sys.stdin)
except Exception:
    sys.exit(0)

session_id = hook_input.get("session_id", "unknown")
transcript_path = hook_input.get("transcript_path", "")

if not transcript_path or not os.path.isfile(transcript_path):
    sys.exit(0)

try:
    entries = []
    with open(transcript_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
except Exception:
    sys.exit(0)

if not entries:
    sys.exit(0)

last_user = ""
tool_names = []
files_edited = set()

for entry in entries[-20:]:
    role = entry.get("role", "")
    content = entry.get("content", "")

    if role == "user":
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    last_user = part["text"][:200]
        elif isinstance(content, str):
            last_user = content[:200]

    elif role == "assistant" and isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                name = part.get("name", "")
                if name:
                    tool_names.append(name)
                inp = part.get("input", {})
                fp = inp.get("file_path", "")
                if fp and name in ("Edit", "Write"):
                    files_edited.add(fp.split("/")[-1])

if not last_user:
    sys.exit(0)

parts = [f"User asked: {last_user}"]
if files_edited:
    parts.append("Files modified: " + ", ".join(sorted(files_edited)[:8]))
if tool_names:
    unique_tools = list(dict.fromkeys(tool_names))[:6]
    parts.append("Tools used: " + ", ".join(unique_tools))

summary = ". ".join(parts)[:500]

if len(summary) < 20:
    sys.exit(0)

payload = json.dumps({
    "text": summary,
    "namespace": "claude-code",
    "source": "claude-code-plugin",
    "salience": 0.7,
    "metadata": {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "session-exchange",
    },
}).encode()

req = urllib.request.Request(
    f"{API_URL}/v1/memory/write",
    data=payload,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    method="POST",
)

try:
    urllib.request.urlopen(req, timeout=8)
except Exception:
    pass

PYEOF

exit 0

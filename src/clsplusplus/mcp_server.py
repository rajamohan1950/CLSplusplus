"""
CLS++ MCP Server — Model Context Protocol for Claude Code, Cursor, Windsurf

Exposes CLS++ memory as MCP tools that LLMs can call natively:
  - recall_memories(query) — retrieve relevant memories
  - store_memory(text) — store a new fact
  - list_sessions() — list active LLM sessions in TRG
  - who_am_i() — return all known facts about the user

No interception, no hooks, no injection. The LLM decides WHEN to recall.

Usage in Claude Code:
  Add to .claude/settings.json:
  {
    "mcpServers": {
      "cls-memory": {
        "command": "python3",
        "args": ["-m", "clsplusplus.mcp_server"],
        "env": {
          "CLS_API_URL": "https://www.clsplusplus.com",
          "CLS_API_KEY": "cls_live_..."
        }
      }
    }
  }

Generate your API key at: https://www.clsplusplus.com/profile.html#keys
Usage in Cursor/Windsurf: same pattern, different config file location.
"""

import json
import os
import sys
import urllib.request
import urllib.error


API_URL = os.environ.get("CLS_API_URL", "https://www.clsplusplus.com")
API_KEY = os.environ.get("CLS_API_KEY", "")


def _api_call(method, endpoint, payload=None):
    """Make API call to CLS++ server."""
    url = f"{API_URL}{endpoint}"
    data = json.dumps(payload).encode() if payload else None
    headers = {
        "Content-Type": "application/json",
    }
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


# ── MCP Tool Definitions ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "recall_memories",
        "description": "Retrieve relevant memories about the user from CLS++ cross-LLM memory. Use this when the user asks about themselves, their preferences, past conversations, or when you need context from their other AI sessions (ChatGPT, Claude, Gemini, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — user identity, preferences, recent work, relationships, etc."
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of memories to return (default 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "store_memory",
        "description": "Store a new fact about the user in CLS++ memory. Use this when the user shares personal information, preferences, decisions, or important context that should persist across AI sessions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The fact to remember — e.g., 'User prefers dark mode', 'User's name is Raj'"
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "who_am_i",
        "description": "Return everything CLS++ knows about the user — identity, preferences, relationships, recent activity. Use this at the start of a conversation or when the user asks 'what do you know about me?'",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
]


# ── MCP Tool Handlers ────────────────────────────────────────────────────

def handle_recall_memories(args):
    query = args.get("query", "")
    limit = args.get("limit", 5)
    result = _api_call("POST", "/v1/memory/read", {
        "query": query,
        "limit": limit,
    })
    items = result.get("items", [])
    if not items:
        return "No memories found for this query."

    lines = []
    for item in items:
        text = item.get("text", "")
        level = item.get("store_level", "?")
        if text.startswith("[Schema:") and len(text.split("]", 1)[-1].strip().split()) < 4:
            continue  # Skip garbage schemas
        lines.append(f"- {text}")

    if not lines:
        return "No relevant memories found."
    return "Memories about this user:\n" + "\n".join(lines)


def handle_store_memory(args):
    text = args.get("text", "").strip()
    if not text:
        return "No text provided to store."
    result = _api_call("POST", "/v1/memory/write", {
        "text": text,
        "source": "mcp",
    })
    if result.get("id"):
        return f"Memory stored: \"{text}\""
    return f"Failed to store memory: {result.get('error', 'unknown error')}"


def handle_who_am_i(args):
    queries = [
        "user identity name who preferences likes dislikes",
        "relationships family friends people",
        "recent work project decisions current status",
        "movies music hobbies interests favorites perfume",
    ]
    all_facts = []
    seen = set()
    for q in queries:
        result = _api_call("POST", "/v1/memory/read", {"query": q, "limit": 5})
        for item in result.get("items", []):
            text = item.get("text", "")
            if text in seen:
                continue
            if text.startswith("[Schema:") and len(text.split("]", 1)[-1].strip().split()) < 4:
                continue
            seen.add(text)
            all_facts.append(text)

    if not all_facts:
        return "No memories stored about this user yet. As they share information, it will be remembered across all AI models."

    lines = ["Here is everything known about this user from their conversations across all AI models:"]
    for f in all_facts:
        lines.append(f"- {f}")
    return "\n".join(lines)


HANDLERS = {
    "recall_memories": handle_recall_memories,
    "store_memory": handle_store_memory,
    "who_am_i": handle_who_am_i,
}


# ── MCP Protocol (JSON-RPC over stdio) ───────────────────────────────────

def handle_request(req):
    method = req.get("method", "")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "cls-memory",
                    "version": "3.0.0",
                },
            },
        }

    if method == "notifications/initialized":
        return None  # No response for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = HANDLERS.get(tool_name)
        if handler:
            try:
                text = handler(arguments)
            except Exception as e:
                text = f"Error: {e}"
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """Run MCP server over stdio (JSON-RPC)."""
    global API_KEY
    if not API_KEY:
        key_file = os.path.expanduser("~/.cls_api_key")
        if os.path.isfile(key_file):
            API_KEY = open(key_file).read().strip()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(req)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

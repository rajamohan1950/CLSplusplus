"""CLS++ CLI — Cross-model memory from your terminal.

Usage:
    cls init                              # Setup API key + LLM keys
    cls chat --model gpt-4o "message"     # Chat with GPT-4 (memories auto-captured)
    cls chat --model claude "message"     # Switch to Claude — it remembers everything
    cls chat                              # Interactive chat session
    cls learn "I prefer dark mode"        # Store a memory
    cls ask "What are my preferences?"    # Query memories
    cls memories                          # List all stored memories
    cls who                               # Auto-generated user profile
    cls status                            # Connection status + memory count
    cls forget "old fact"                 # Delete a memory
    cls absorb file.txt                   # Bulk-learn from a file
    cls context "coding help"             # Get LLM-ready context string
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path.home() / ".clsplusplus"
DEFAULT_URL = "https://www.clsplusplus.com"

# Model routing
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "o1", "o1-mini"}
ANTHROPIC_MODELS = {"claude", "claude-sonnet", "claude-haiku", "claude-opus",
                    "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001"}
GEMINI_MODELS = {"gemini", "gemini-pro", "gemini-2.0-flash"}


def _load_config() -> dict:
    """Load config from ~/.clsplusplus."""
    cfg = {
        "api_key": os.environ.get("CLS_API_KEY", ""),
        "user": os.environ.get("CLS_USER", os.environ.get("USER", "default")),
        "url": os.environ.get("CLS_BASE_URL", DEFAULT_URL),
        "openai_key": os.environ.get("OPENAI_API_KEY", ""),
        "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "google_key": os.environ.get("GOOGLE_API_KEY", ""),
    }
    if CONFIG_PATH.exists():
        cp = configparser.ConfigParser()
        cp.read(CONFIG_PATH)
        if cp.has_section("default"):
            cfg["api_key"] = cp.get("default", "api_key", fallback=cfg["api_key"])
            cfg["user"] = cp.get("default", "user", fallback=cfg["user"])
            cfg["url"] = cp.get("default", "url", fallback=cfg["url"])
        if cp.has_section("keys"):
            cfg["openai_key"] = cp.get("keys", "openai", fallback=cfg["openai_key"])
            cfg["anthropic_key"] = cp.get("keys", "anthropic", fallback=cfg["anthropic_key"])
            cfg["google_key"] = cp.get("keys", "google", fallback=cfg["google_key"])
    return cfg


def _save_config(cfg: dict) -> None:
    """Save config to ~/.clsplusplus."""
    cp = configparser.ConfigParser()
    cp["default"] = {
        "api_key": cfg.get("api_key", ""),
        "user": cfg.get("user", ""),
        "url": cfg.get("url", DEFAULT_URL),
    }
    keys = {}
    if cfg.get("openai_key"):
        keys["openai"] = cfg["openai_key"]
    if cfg.get("anthropic_key"):
        keys["anthropic"] = cfg["anthropic_key"]
    if cfg.get("google_key"):
        keys["google"] = cfg["google_key"]
    if keys:
        cp["keys"] = keys
    with open(CONFIG_PATH, "w") as f:
        cp.write(f)
    print(f"Config saved to {CONFIG_PATH}")


def _brain(cfg: dict):
    """Create a Brain instance from config."""
    from clsplusplus import Brain
    return Brain(user=cfg["user"], api_key=cfg["api_key"], url=cfg["url"])


def _resolve_model(model: str) -> str:
    """Resolve short model names to full IDs."""
    aliases = {
        "gpt4": "gpt-4o",
        "gpt": "gpt-4o",
        "openai": "gpt-4o",
        "claude": "claude-sonnet-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-5-20251001",
        "opus": "claude-opus-4-20250514",
        "gemini": "gemini-2.0-flash",
    }
    return aliases.get(model.lower(), model)


def _model_provider(model: str) -> str:
    """Determine provider from model name."""
    m = model.lower()
    if any(m.startswith(p) for p in ("gpt", "o1")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    return "openai"  # default


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(args, cfg):
    """Interactive setup."""
    print("CLS++ Setup")
    print("=" * 40)

    # API key
    key = args.key or cfg["api_key"]
    if not key:
        key = input("CLS++ API key (from clsplusplus.com/profile.html#keys): ").strip()
    if not key:
        print("No API key provided. Get one at https://www.clsplusplus.com/profile.html#keys")
        return

    # User
    user = args.user or cfg["user"]
    if user == os.environ.get("USER", "default"):
        entered = input(f"Username [{user}]: ").strip()
        if entered:
            user = entered

    # LLM keys
    openai_key = args.openai_key or cfg["openai_key"]
    if not openai_key:
        openai_key = input("OpenAI API key (optional, press Enter to skip): ").strip()

    anthropic_key = args.anthropic_key or cfg["anthropic_key"]
    if not anthropic_key:
        anthropic_key = input("Anthropic API key (optional, press Enter to skip): ").strip()

    google_key = args.google_key or cfg["google_key"]
    if not google_key:
        google_key = input("Google API key (optional, press Enter to skip): ").strip()

    new_cfg = {
        "api_key": key,
        "user": user,
        "url": args.url or cfg["url"],
        "openai_key": openai_key,
        "anthropic_key": anthropic_key,
        "google_key": google_key,
    }

    # Test connection
    print("\nTesting connection...")
    try:
        import httpx
        resp = httpx.get(f"{new_cfg['url'].rstrip('/')}/v1/health", timeout=10)
        if resp.status_code == 200:
            print(f"Connected to {new_cfg['url']}.")
        else:
            print(f"Server responded with {resp.status_code}. Check your URL.")
    except Exception as e:
        print(f"Warning: could not reach server ({e}). Config saved anyway.")

    _save_config(new_cfg)
    print("\nReady! Try: cls learn \"I prefer Python\"")


def cmd_chat(args, cfg):
    """Chat with any LLM — memories auto-captured and injected."""
    import httpx

    model = _resolve_model(args.model or "gpt-4o")
    provider = _model_provider(model)
    url = cfg["url"].rstrip("/")

    # Determine LLM API key
    if provider == "openai":
        llm_key = cfg.get("openai_key", "")
    elif provider == "anthropic":
        llm_key = cfg.get("anthropic_key", "")
    else:
        llm_key = cfg.get("google_key", "")

    if not llm_key:
        print(f"No {provider} API key configured. Run: cls init")
        return

    def send_message(message: str) -> str:
        """Send a message through CLS++ proxy and return response."""
        headers = {"Content-Type": "application/json"}

        if provider == "anthropic":
            # Anthropic format
            headers["x-api-key"] = llm_key
            headers["anthropic-version"] = "2023-06-01"
            if cfg["api_key"]:
                headers["Authorization"] = f"Bearer {cfg['api_key']}"
            body = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": message}],
            }
            endpoint = f"{url}/v1/messages"
        else:
            # OpenAI format (also works for Gemini via compatible API)
            headers["Authorization"] = f"Bearer {llm_key}"
            body = {
                "model": model,
                "messages": [{"role": "user", "content": message}],
            }
            endpoint = f"{url}/v1/chat/completions"

        # Add user namespace
        body["user"] = cfg["user"]

        resp = httpx.post(endpoint, json=body, headers=headers, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        if provider == "anthropic":
            return data.get("content", [{}])[0].get("text", "")
        else:
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if args.message:
        # One-shot mode
        message = " ".join(args.message)
        try:
            reply = send_message(message)
            print(f"\n{reply}\n")
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code} — {e.response.text[:200]}")
        except Exception as e:
            print(f"Error: {e}")
        return

    # Interactive mode
    print(f"CLS++ Chat (model: {model})")
    print("Type /quit to exit, /memories to list, /who for profile\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue
        if message == "/quit":
            print("Goodbye!")
            break
        if message == "/memories":
            brain = _brain(cfg)
            for m in brain.all():
                print(f"  - {m}")
            continue
        if message == "/who":
            brain = _brain(cfg)
            profile = brain.who()
            print(f"\n{profile.get('summary', 'No profile yet.')}\n")
            continue

        try:
            reply = send_message(message)
            print(f"\nAI: {reply}\n")
        except httpx.HTTPStatusError as e:
            print(f"Error: {e.response.status_code}")
        except Exception as e:
            print(f"Error: {e}")


def cmd_learn(args, cfg):
    """Store a memory."""
    brain = _brain(cfg)
    fact = " ".join(args.fact)
    mid = brain.learn(fact)
    print(f"Learned: \"{fact}\"")


def cmd_ask(args, cfg):
    """Query memories."""
    brain = _brain(cfg)
    question = " ".join(args.question)
    results = brain.ask(question, limit=args.limit)
    if not results:
        print("No matching memories found.")
        return
    for r in results:
        print(f"  - {r}")


def cmd_memories(args, cfg):
    """List all stored memories."""
    brain = _brain(cfg)
    mems = brain.all(limit=args.limit)
    if not mems:
        print("No memories stored yet. Try: cls learn \"I prefer Python\"")
        return
    print(f"{len(mems)} memories:\n")
    for i, m in enumerate(mems, 1):
        print(f"  {i}. {m}")


def cmd_who(args, cfg):
    """Show auto-generated user profile."""
    brain = _brain(cfg)
    profile = brain.who()
    print(f"\nProfile: {profile.get('user', cfg['user'])}")
    print(f"Memories: {profile.get('count', 0)}\n")
    if profile.get("summary"):
        print(profile["summary"])
    print()
    for fact in profile.get("facts", []):
        print(f"  - {fact}")


def cmd_status(args, cfg):
    """Show connection status and memory count."""
    import httpx

    url = cfg["url"].rstrip("/")
    print(f"Server:  {url}")
    print(f"User:    {cfg['user']}")
    print(f"API Key: {'configured' if cfg['api_key'] else 'not set'}")
    print(f"Config:  {CONFIG_PATH}")

    # Health check
    try:
        resp = httpx.get(f"{url}/v1/health", timeout=10)
        data = resp.json()
        print(f"Status:  {data.get('status', 'unknown')}")
    except Exception:
        print("Status:  unreachable")

    # Memory count
    if cfg["api_key"]:
        try:
            brain = _brain(cfg)
            print(f"Memories: {brain.count()}")
        except Exception:
            print("Memories: unable to fetch")

    # LLM keys
    print(f"\nOpenAI:    {'configured' if cfg.get('openai_key') else 'not set'}")
    print(f"Anthropic: {'configured' if cfg.get('anthropic_key') else 'not set'}")
    print(f"Google:    {'configured' if cfg.get('google_key') else 'not set'}")


def cmd_forget(args, cfg):
    """Delete a memory."""
    brain = _brain(cfg)
    fact = " ".join(args.fact)
    ok = brain.forget(fact)
    if ok:
        print(f"Forgotten: \"{fact}\"")
    else:
        print(f"Could not find memory: \"{fact}\"")


def cmd_absorb(args, cfg):
    """Bulk-learn from a file."""
    brain = _brain(cfg)
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return
    content = path.read_text()
    count = brain.absorb(content, source=path.name)
    print(f"Absorbed {count} facts from {path.name}")


def cmd_context(args, cfg):
    """Get LLM-ready context string."""
    brain = _brain(cfg)
    topic = " ".join(args.topic) if args.topic else ""
    ctx = brain.context(topic)
    if ctx:
        print(ctx)
    else:
        print("No relevant context found.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cls",
        description="CLS++ — Persistent memory across every AI model.",
        epilog="Get started: cls init",
    )
    parser.add_argument("--user", help="User identifier (overrides config)")
    parser.add_argument("--key", help="CLS++ API key (overrides config)")
    parser.add_argument("--url", help="Server URL (overrides config)")

    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Setup API keys and configuration")
    p_init.add_argument("--key", dest="key", help="CLS++ API key")
    p_init.add_argument("--user", help="Username")
    p_init.add_argument("--url", help="Server URL")
    p_init.add_argument("--openai-key", dest="openai_key", help="OpenAI API key")
    p_init.add_argument("--anthropic-key", dest="anthropic_key", help="Anthropic API key")
    p_init.add_argument("--google-key", dest="google_key", help="Google API key")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with any LLM — memories transfer across models")
    p_chat.add_argument("--model", "-m", default="gpt-4o", help="Model: gpt-4o, claude, gemini (default: gpt-4o)")
    p_chat.add_argument("message", nargs="*", help="Message (omit for interactive mode)")

    # learn
    p_learn = sub.add_parser("learn", help="Store a memory")
    p_learn.add_argument("fact", nargs="+", help="Fact to remember")

    # ask
    p_ask = sub.add_parser("ask", help="Query memories")
    p_ask.add_argument("question", nargs="+", help="Question to ask")
    p_ask.add_argument("--limit", "-n", type=int, default=5, help="Max results (default: 5)")

    # memories
    p_mems = sub.add_parser("memories", help="List all stored memories")
    p_mems.add_argument("--limit", "-n", type=int, default=50, help="Max results (default: 50)")

    # who
    sub.add_parser("who", help="Show auto-generated user profile")

    # status
    sub.add_parser("status", help="Show connection status and memory count")

    # forget
    p_forget = sub.add_parser("forget", help="Delete a memory")
    p_forget.add_argument("fact", nargs="+", help="Fact to forget")

    # absorb
    p_absorb = sub.add_parser("absorb", help="Bulk-learn from a file")
    p_absorb.add_argument("file", help="Path to text file")

    # context
    p_context = sub.add_parser("context", help="Get LLM-ready context string")
    p_context.add_argument("topic", nargs="*", help="Topic to get context for")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Load config, apply CLI overrides
    cfg = _load_config()
    if getattr(args, "key", None) and args.command != "init":
        cfg["api_key"] = args.key
    if getattr(args, "user", None) and args.command != "init":
        cfg["user"] = args.user
    if getattr(args, "url", None) and args.command != "init":
        cfg["url"] = args.url

    commands = {
        "init": cmd_init,
        "chat": cmd_chat,
        "learn": cmd_learn,
        "ask": cmd_ask,
        "memories": cmd_memories,
        "who": cmd_who,
        "status": cmd_status,
        "forget": cmd_forget,
        "absorb": cmd_absorb,
        "context": cmd_context,
    }

    fn = commands.get(args.command)
    if fn:
        fn(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
